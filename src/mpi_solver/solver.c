#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"

// значение 0 - синхронные посылки
// значение 1 - асинхронные посылки
#define ASYNC 0

#define MIN_COL_COUNT 1
#define MIN_ROW_COUNT 1
#define MAX_COL_COUNT 10000
#define MAX_ROW_COUNT 10000

#define EPSILON 0.000000000001
#define A_PARAM 1
#define B_PARAM 2
#define X_PARAM 3

typedef struct 
{
	int ARowCount;
	int AColCount;
	int BRowCount;
	int BColCount;
} TMulInfo;

typedef struct
{
	int RowCount;
	int ColCount;
} TMatrixSize;

typedef struct
{
	double* Ptr;
	int ColCount;
	int FromRow;
	int ToRow;
	int FromCol;
	int ToCol;
} TCopyInfo;

static
char InvMatrixError[] = "InvMatrixError";

static
void AllocMatrix(double* Matrix, const int Size)
{
	Matrix = (double*) malloc(sizeof(double)*Size*Size);
	if (Matrix == NULL) perror("");
	//return Matrix;
}

// поверяет значение Val
// на принадлежность отрезку [MinVal, MaxVal]
// возвращает 1 в случае успеха
// возаращает 0 иначе
static
int CheckVal(const int Val,
			 const int MinVal,
			 const int MaxVal,
			 const char* ValErrStr)
{
	if (Val < MinVal)
	{
		printf("%s is too small\n",ValErrStr);
		return 0;
	}
	if (Val > MaxVal)
	{
		printf("%s is too big\n",ValErrStr);
		return 0;
	}
	return 1;
}

// читает размер матрицы из _File в MatrixSize
// FileName - имя файла
// MatrixSize - указатель на заполняемую структуру
// MaxRowCount - максимальное число строк
// MaxColCount - максимальное число столбцов
static
int ReadMatrixSize(FILE* _File,
				   TMatrixSize* MatrixSize,
				   const int MaxRowCount,
				   const int MaxColCount)
{
	// чтение RowCount
	if (fscanf(_File,"%i",&MatrixSize->RowCount) < 1)
	{
		perror("Error reading RowCount");
		return 0;
	}
	// проверка значение RowCount
	if (!CheckVal(MatrixSize->RowCount,
				  MIN_ROW_COUNT,
				  MaxRowCount,
				  "RowCount"))
		return 0;	
	// чтение ColCount
	if (fscanf(_File,"%i",&MatrixSize->ColCount) < 1)
	{
		perror("Error reading ColCount");
		return 0;
	}
	// проверка значения ColCount
	if (!CheckVal(MatrixSize->ColCount,
				  MIN_COL_COUNT,
				  MaxColCount,
				  "ColCount"))
		return 0;
	return 1;
}

// читает матрицу Matrix из файла _File
// _File - дескриптор файла - потока
// Matrix - указатель на непрерывную область памяти,
// куда ляжет матрица (строка за строкой)
// MatrixSize - указатель на структуру с размером матрицы
// возвращает 1 в случае успеха
// возвращает 0 в случае ошибки
static
int ReadMatrix(FILE* _File,
			   double* Matrix,
			   const TMatrixSize* MatrixSize)
{
	int Row;
	int Col;
	for (Row = 0; Row < MatrixSize->RowCount; Row++)
	{
		for (Col = 0; Col < MatrixSize->ColCount; Col++)
		{
			// чтение элемента
			if (fscanf(_File,"%lf",&Matrix[Row*MatrixSize->ColCount + Col]) < 1)
			{
				return 0;
			}
		}
	}
	return 1;
}

// Matrix3 = Matrix1 + Matrix2
// матрицы - квадратные
static
void AddMatrix(double* Matrix1,
			   double* Matrix2,
			   double* Matrix3,
			   const int Size)
{
	int Row;
	int Col;
	for (Row = 0; Row < Size; Row++)
		for (Col = 0; Col < Size; Col++)
			Matrix3[Row*Size+Col] = 
				Matrix1[Row*Size+Col] +
					Matrix2[Row*Size+Col];
}

//Matrix3 = Matrix1 - Matrix2
// матрицы - квадратные
static
void SubMatrix(double* Matrix1,
			   double* Matrix2,
			   double* Matrix3,
			   const int Size)
{
	int Row;
	int Col;
	for (Row = 0; Row < Size; Row++)
		for (Col = 0; Col < Size; Col++)
			Matrix3[Row*Size+Col] = 
				Matrix1[Row*Size+Col] -
					Matrix2[Row*Size+Col];
}

// Matrix3 = Matrix1*Matrix2
static
void MulMatrix(double* Matrix1,
			   const int Matrix1RowCount,
			   const int Matrix1ColCount,
			   double* Matrix2,
			   const int Matrix2RowCount,
			   const int Matrix2ColCount,
			   double* Matrix3)
{
	if (Matrix1ColCount != Matrix2RowCount) return;
	int i;
	int j;
	int r;
	for (i = 0; i < Matrix1RowCount; i++)
		for (j=0;j < Matrix2ColCount; j++)
		{
			Matrix3[i*Matrix2ColCount+j] = 0.0;
			for (r = 0; r < Matrix1ColCount; r++)
				Matrix3[i*Matrix2ColCount+j]+=
					Matrix1[i*Matrix1ColCount+r] * Matrix2[r*Matrix2ColCount+j];
		}
}

// CMatrix = AMatrix*BMatrix
// ProcCount - общее число параллельных процессов
static
void ParallelMulMatrix(double* AMatrix,
					   const int ARowCount,
					   const int AColCount,
					   double* BMatrix,
					   const int BRowCount,
					   const int BColCount,
					   double* CMatrix,
					   const int ProcCount)
{
	int i;
	int Continue = 0;
	int ARowCountForRoot;
	int ARowCountForSlave;
	int SlavePtr;
	TMulInfo MulInfo;
	#if ASYNC==1
	static
	MPI_Request ReqGetC[256];
	MPI_Request ReqSendB;
	#endif
	static
	MPI_Status StatGetC[256];
	
	// вычисляем, сколько строк матрицы A посылать ведомым
	ARowCountForSlave = ARowCount / ProcCount;
	ARowCountForRoot = ARowCount - (ARowCountForSlave*(ProcCount-1));	
	MulInfo.AColCount = AColCount;
	MulInfo.ARowCount = ARowCountForSlave;
	MulInfo.BColCount = BColCount;
	MulInfo.BRowCount = BRowCount;
	// если число процессов > строк матрицы A то все делаем на ведущем
	if (ARowCountForRoot != ARowCount)
	{
		// запускаем ведомые
		Continue = 1;
		MPI_Bcast(&Continue,1,MPI_INT,0,MPI_COMM_WORLD);
		// отсылаем размеры умножаемых матриц
		MPI_Bcast(&MulInfo,sizeof(MulInfo),MPI_CHAR,0,MPI_COMM_WORLD);
		// отсылаем матрицу B всем процессам
		for (i = 1; i < ProcCount; i++)
			#if ASYNC==1
			MPI_Isend(BMatrix,//buf
					  MulInfo.BRowCount*MulInfo.BColCount, // cnt
					  MPI_DOUBLE,
					  i, //dest
					  0,//tag
					  MPI_COMM_WORLD,
					  &ReqSendB);
			#else
			MPI_Send(BMatrix,MulInfo.BRowCount*MulInfo.BColCount,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
			#endif
		// отпарвляем строки матрицы A ведомым
		// первые строки обрабатывает root
		SlavePtr = ARowCountForRoot*MulInfo.AColCount;
		for (i = 1; i < ProcCount; i++)
		{
			#if ASYNC==1
			MPI_Isend(&AMatrix[SlavePtr],//buf
					  MulInfo.ARowCount*MulInfo.AColCount, // cnt
					  MPI_DOUBLE,
					  i, //dest
					  0,//tag
					  MPI_COMM_WORLD,
					  &ReqSendB);
			#else
			MPI_Send(&AMatrix[SlavePtr],MulInfo.ARowCount*MulInfo.AColCount,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
			#endif
			SlavePtr+= MulInfo.ARowCount*MulInfo.AColCount;
		}
		// вычисляем свою часть матрицы C
		MulMatrix(AMatrix,
				  ARowCountForRoot,
				  MulInfo.AColCount,
				  BMatrix,
			      MulInfo.BRowCount,
			      MulInfo.BColCount,
			      CMatrix);
		// начинаем прием элементов матрицы C
		SlavePtr = ARowCountForRoot*MulInfo.BColCount;
		for (i = 1; i < ProcCount; i++)
		{
			#if ASYNC==1
			MPI_Irecv(&CMatrix[SlavePtr], //buf
					  MulInfo.ARowCount*MulInfo.BColCount,//count
					  MPI_DOUBLE,//type
					  i, //src
					  0, //tag
					  MPI_COMM_WORLD,
					  &ReqGetC[i-1]);
			#else
			MPI_Recv(&CMatrix[SlavePtr],MulInfo.ARowCount*MulInfo.BColCount,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&StatGetC[i-1]);
			#endif
			SlavePtr+= MulInfo.ARowCount*MulInfo.BColCount;
		}
		#if ASYNC==1		  
		MPI_Waitall(ProcCount-1,ReqGetC,StatGetC);
		#endif
	}
	else
	{
		MulMatrix(AMatrix,
				  ARowCountForRoot,
				  MulInfo.AColCount,
				  BMatrix,
			      MulInfo.BRowCount,
			      MulInfo.BColCount,
			      CMatrix);
	}
}

// Инвертирует знак матрицы Matrix1
// помещает результат в матрицу Matrix2
static
void MatrixInvSign(double* Matrix1,
				   double* Matrix2,
				   const int RowCount,
				   const int ColCount)
{
	int Row;
	int Col;
	for (Row = 0; Row < RowCount; Row++)
		for (Col = 0; Col < ColCount; Col++)
			Matrix2[Row*ColCount+Col]= -Matrix1[Row*ColCount+Col];
}

// копирует матрицу, описанную структурой
// Src
// в матрицу, описанную структурой Dst
static
void CopyMatrix(TCopyInfo* Src,
				TCopyInfo* Dst)
{
	int SrcRow;
	int SrcCol;
	int DstRow = Dst->FromRow;
	int DstCol;
	for (SrcRow = Src->FromRow; (SrcRow <= Src->ToRow) && (DstRow <= Dst->ToRow); SrcRow++)
	{
		DstCol = Dst->FromCol;
		for (SrcCol = Src->FromCol; (SrcCol <= Src->ToCol) && (DstCol <= Dst->ToCol); SrcCol++)
		{
			Dst->Ptr[DstRow*Dst->ColCount + DstCol] = 
				Src->Ptr[SrcRow*Src->ColCount + SrcCol];
			DstCol++;
		}
		DstRow++;
	}
}

static
int ParallelGetInvMatrix(double* Matrix,
						  const int Size,
						  double* InvMatrix,
						  const int ProcCount)
{
	double* A;
	double* InvA;
	double* B;
	double* C;
	double* D;
	double* InvH;
	double* Temp;
	int n;
	int q;
	int res;
	TCopyInfo Src;
	TCopyInfo Dst;
	if (Size > 1)
	{
		// разделяем входную матрицу на 4 примерно равнае части
		q = Size/2;
		n = Size - q;
		// выделяем память под вспомогательные матрицы
		A = (double*)malloc(sizeof(double)*n*n);
		if (A == NULL)
		{
			perror(InvMatrixError);
			return 0;
		}
		InvA = (double*)malloc(sizeof(double)*n*n);
		if (InvA == NULL)
		{
			perror(InvMatrixError);
			free(A);
			return 0;
		}
		B = (double*)malloc(sizeof(double)*n*q);
		if (B == NULL)
		{
			perror(InvMatrixError);
			free(A);
			free(InvA);
			return 0;
		}
		C = (double*)malloc(sizeof(double)*q*n);
		if (C == NULL)
		{
			perror(InvMatrixError);
			free(A);
			free(InvA);
			free(B);
			return 0;
		}
		D = (double*)malloc(sizeof(double)*q*q);
		if (D == NULL)
		{
			perror(InvMatrixError);
			free(A);
			free(InvA);
			free(B);
			free(C);
			return 0;
		}
		InvH = (double*)malloc(sizeof(double)*q*q);
		if (InvH == NULL)
		{
			perror(InvMatrixError);
			free(A);
			free(InvA);
			free(B);
			free(C);
			free(D);
			return 0;
		}
		Temp = (double*)malloc(sizeof(double)*n*n);
		if (Temp == NULL)
		{
			perror(InvMatrixError);
			free(A);
			free(InvA);
			free(B);
			free(C);
			free(D);
			free(InvH);
			return 0;
		}
		// копируем подматрицы
		// копируем A
		Src.Ptr = Matrix;
		Src.ColCount = Size;
		Src.FromRow = 0;
		Src.FromCol = 0;
		Src.ToRow = n-1;
		Src.ToCol = n-1;
		Dst.Ptr = A;
		Dst.ColCount = n;
		Dst.FromRow = 0;
		Dst.FromCol = 0;
		Dst.ToRow = n-1;
		Dst.ToCol = n-1;
		CopyMatrix(&Src,&Dst);
		// копируем B
		Src.FromCol = n;
		Src.ToCol = Size-1;
		Dst.Ptr = B;
		Dst.ColCount = q;
		Dst.ToCol = q-1;
		CopyMatrix(&Src,&Dst);
		// копируем C
		Src.FromRow = n;
		Src.ToRow = Size-1;
		Src.FromCol = 0;
		Src.ToCol = n-1;
		Dst.Ptr = C;
		Dst.ColCount = n;
		Dst.ToRow = q-1;
		Dst.ToCol = n-1;
		CopyMatrix(&Src,&Dst);
		// копируем D
		Src.FromCol = n;
		Src.ToCol = Size - 1;
		Dst.Ptr = D;
		Dst.ColCount = q;
		Dst.ToRow = q-1;
		Dst.ToCol = q-1;
		CopyMatrix(&Src,&Dst);
		// вычисляем обратную A
		res = ParallelGetInvMatrix(A,n,InvA,ProcCount);
		if (res)
		{
			// вычисляем обратную H
			ParallelMulMatrix(C,q,n,InvA,n,n,A,ProcCount); // A = C*(A^-1)
			ParallelMulMatrix(A,q,n,B,n,q,InvH,ProcCount); // InvH = A*B
			SubMatrix(D,InvH,A,q); // A = D - InvH
			res = ParallelGetInvMatrix(A,q,InvH,ProcCount); // InvH = A^-1
			if (res)
			{
				
				// вычисляем InvMatrix[0][1]
				ParallelMulMatrix(InvA,n,n,B,n,q,A,ProcCount); // A(n,q)
				ParallelMulMatrix(A,n,q,InvH,q,q,Temp,ProcCount); // Temp(n,q)
				MatrixInvSign(Temp,A,n,q); // A(n,q)
				// копируем
				Src.Ptr = A;
				Src.ColCount = q;
				Src.FromRow = 0;
				Src.FromCol = 0;
				Src.ToRow = n-1;
				Src.ToCol = q-1;
				Dst.Ptr = InvMatrix;
				Dst.ColCount = Size;
				Dst.FromRow = 0;
				Dst.FromCol = n;
				Dst.ToRow = n-1;
				Dst.ToCol = Size-1;
				CopyMatrix(&Src,&Dst);
				// вычисляем InvMatrix[0][0]
				ParallelMulMatrix(Temp,n,q,C,q,n,A,ProcCount); // A(n,n)
				ParallelMulMatrix(A,n,n,InvA,n,n,Temp,ProcCount); // Temp(n,n)
				AddMatrix(InvA,Temp,A,n); //A(n,n)
				// копируем
				Src.Ptr = A;
				Src.ColCount = n;
				Src.FromRow = 0;
				Src.FromCol = 0;
				Src.ToRow = n-1;
				Src.ToCol = n-1;
				Dst.Ptr = InvMatrix;
				Dst.ColCount = Size;
				Dst.FromRow = 0;
				Dst.FromCol = 0;
				Dst.ToRow = n-1;
				Dst.ToCol = n-1;
				CopyMatrix(&Src,&Dst);
				// вычисляем InvMatrix[1][0]
				ParallelMulMatrix(InvH,q,q,C,q,n,A,ProcCount); //A(q,n)
				ParallelMulMatrix(A,q,n,InvA,n,n,Temp,ProcCount); //Temp(q,n)
				MatrixInvSign(Temp,A,q,n); //A (q,n)
				// копируем
				Src.Ptr = A;
				Src.ColCount = n;
				Src.FromRow = 0;
				Src.FromCol = 0;
				Src.ToRow = q-1;
				Src.ToCol = n-1;
				Dst.Ptr = InvMatrix;
				Dst.ColCount = Size;
				Dst.FromRow = n;
				Dst.FromCol = 0;
				Dst.ToRow = Size-1;
				Dst.ToCol = n-1;
				CopyMatrix(&Src,&Dst);
				// копирируем InvMatrix[1][1]
				Src.Ptr = InvH;
				Src.ColCount = q;
				Src.FromRow = 0;
				Src.FromCol = 0;
				Src.ToRow = q-1;
				Src.ToCol = q-1;
				Dst.Ptr = InvMatrix;
				Dst.ColCount = Size;
				Dst.FromRow = n;
				Dst.FromCol = n;
				Dst.ToRow = Size-1;
				Dst.ToCol = Size-1;
				CopyMatrix(&Src,&Dst);
			}
		}
		free(A);
		free(InvA);
		free(B);
		free(C);
		free(D);
		free(InvH);
		free(Temp);
		return res;
	}
	else
	{
		if (fabs(Matrix[0]) > EPSILON)
		{
			InvMatrix[0] = 1.0 / Matrix[0];
			return 1;
		}
		else return 0;
	}
}

// фоновая функция ведомых
static
void SlaveProc(const int rank, const int size)
{
	TMulInfo MulInfo;
	#if ASYNC==1
	MPI_Request ReqGetA;
	MPI_Request ReqGetB;
	MPI_Request ReqSendC;
	MPI_Status StatSendC;
	#endif
	MPI_Status StatGetA;
	MPI_Status StatGetB;
	
	int Continue = 1;
	int MaxMatrixSize;
	// CMatrix = AMatrix*BMatrix;
	double* BMatrix = NULL;
	double* AMatrix = NULL;
	double* CMatrix = NULL;
	// ожидание приема максимального размера матрицы от корня
	MPI_Bcast(&MaxMatrixSize,1,MPI_INT,0,MPI_COMM_WORLD);
	AllocMatrix(AMatrix,MaxMatrixSize);
	// выделение памяти
	if (AMatrix == NULL)
		MPI_Abort(MPI_COMM_WORLD,1);
	AllocMatrix(BMatrix,MaxMatrixSize)
	if (BMatrix == NULL)
	{
		free(AMatrix);
		MPI_Abort(MPI_COMM_WORLD,1);
	}

	AllocMatrix(CMatrix,MaxMatrixSize);
	if (CMatrix == NULL)
	{
		free(AMatrix);
		free(BMatrix);
		MPI_Abort(MPI_COMM_WORLD,1);
	}
	// цикл умножения (получение строки матрицы C)
	while(Continue)
	{
		// принимаем информацию о необходимости завершения
		MPI_Bcast(&Continue,1,MPI_INT,0,MPI_COMM_WORLD);
		if (Continue)
		{
			// принимаем информацию о подлежащих умножению матрицах
			MPI_Bcast(&MulInfo,sizeof(MulInfo),MPI_CHAR,0,MPI_COMM_WORLD);
			// принимаем матрицу B
			#if ASYNC==1
			// принимаем матрицу B
			MPI_Irecv(BMatrix, //buf
					  MulInfo.BColCount*MulInfo.BRowCount,//count
					  MPI_DOUBLE,//type
					  0, //src
					  0, //tag
					  MPI_COMM_WORLD,
					  &ReqGetB);
			// принимаем матрицу A
			MPI_Irecv(AMatrix, //buf
					  MulInfo.ARowCount*MulInfo.AColCount,//count
					  MPI_DOUBLE,//type
					  0, //src
					  0, //tag
					  MPI_COMM_WORLD,
					  &ReqGetA);
			MPI_Wait(&ReqGetB,&StatGetB);
			MPI_Wait(&ReqGetA,&StatGetA);
			#else
			// принимаем матрицу B
			MPI_Recv(BMatrix,MulInfo.BColCount*MulInfo.BRowCount,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&StatGetB);
			// принимаем матрицу A
			MPI_Recv(AMatrix,MulInfo.ARowCount*MulInfo.AColCount,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&StatGetA);
			// CMatrix = AMatrix*BMatrix
			#endif
			MulMatrix(AMatrix,
					  MulInfo.ARowCount,
					  MulInfo.AColCount,
					  BMatrix,
					  MulInfo.BRowCount,
					  MulInfo.BColCount,
					  CMatrix);
			#if ASYNC==1
			// отсылаем CMatrix к ведущему
			MPI_Isend(CMatrix,//buf
					  MulInfo.ARowCount*MulInfo.BColCount,//cnt
					  MPI_DOUBLE,//type
					  0,//dest
					  0,//tag
					  MPI_COMM_WORLD,
					  &ReqSendC);
			MPI_Wait(&ReqSendC,&StatSendC);
			#else
			// отсылаем CMatrix к ведущему
			MPI_Send(CMatrix,MulInfo.ARowCount*MulInfo.BColCount,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
			#endif
		}
	}
	free(AMatrix);
	free(BMatrix);
	free(CMatrix);
}

int main(int argc, char ** argv)
{
	double Time;
	int rank; // номер потока
	int size; // всего потоков
	int Continue;
	int i;
	FILE* AFile;
	FILE* BFile;
	FILE* XFile;
	TMatrixSize AMatrixSize;
	TMatrixSize BMatrixSize;
	double* A; // матрица коэфициентов (матрица A)
	double* B; // вектор свободных членов (вектор B)
	double* X; // вектор решения
	MPI_Init (&argc, &argv);       
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);    
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	if (rank == 0)
	{
		puts("Block method for solving systems of linear equations. Multiple process mode.");
		printf("Started %d parallel processes\n",size);		
		// проверка аргуметов
		if (argc < 4)
		{
			printf("Error: not enough actual parameters\n");
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		// открытие файла с матрицей A
		if ((AFile = fopen(argv[A_PARAM],"r")) == NULL)
		{
			perror(argv[A_PARAM]);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		printf("Reading matrix A...\n");
		// чтение размера матрицы A
		if (!ReadMatrixSize(AFile,
							&AMatrixSize,
							MAX_ROW_COUNT,
							MAX_COL_COUNT))
		{
			printf("Error: bad Matrix A size\n");
			fclose(AFile);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		// проверить квадратная ли матрица
		if (AMatrixSize.ColCount != AMatrixSize.RowCount)
		{
			printf("Error: matrix A is not square\n");
			fclose(AFile);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		// выделение памяти под матрицу A
		A = (double*)malloc(sizeof(double)*AMatrixSize.ColCount*AMatrixSize.RowCount);
		if (A == NULL)
		{
			perror("Can not allocate matrix A\n");
			fclose(AFile);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		// чтение матрицы A из файла
		if (!ReadMatrix(AFile,
						A,
						&AMatrixSize))
		{
			printf("Error reading matrix A\n");
			free(A);
			fclose(AFile);
			MPI_Abort(MPI_COMM_WORLD,1);	
		}
		// открытие файла с вектором B
		if ((BFile = fopen(argv[B_PARAM],"r")) == NULL)
		{
			perror(argv[B_PARAM]);
			free(A);
			fclose(AFile);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		printf("Reading vector B...\n");
		// чтение размера вектора B
		if (!ReadMatrixSize(BFile,
							&BMatrixSize,
							AMatrixSize.RowCount,
							1))
		{
			printf("Error: bad B vector size\n");
			free(A);
			fclose(AFile);
			fclose(BFile);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		// ветор B должен быть соразмерен матрице A
		if (BMatrixSize.RowCount != AMatrixSize.RowCount)
		{
			printf("Error: vector B size is not equal to the matrix size\n");
			free(A);
			fclose(AFile);
			fclose(BFile);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		// выделение памяти под вектор B
		B = (double*)malloc(sizeof(double)*BMatrixSize.RowCount);
		if (B == NULL)
		{
			perror("Can not allocate vector B\n");
			free(A);
			fclose(AFile);
			fclose(BFile);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		// чтение вектора B
		if (!ReadMatrix(BFile,
						B,
						&BMatrixSize))
		{
			printf("Error reading vector B\n");
			free(A);
			fclose(AFile);
			fclose(BFile);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		// создание файла результата
		if ((XFile = fopen(argv[X_PARAM],"w")) == NULL)
		{
			perror(argv[B_PARAM]);
			free(A);
			free(B);
			fclose(AFile);
			fclose(BFile);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		// выделение памяти под вектор решения
		X = (double*)malloc(sizeof(double)*BMatrixSize.RowCount);
		if (X == NULL)
		{
			perror("Can not allocate vector X");
			free(A);
			free(B);
			fclose(AFile);
			fclose(BFile);
			fclose(XFile);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		printf("System of %d variables\n",BMatrixSize.RowCount);
		printf("Solving the system...\n");
		// инициализация ведомых (отсылаем максимальный размер матриц)
		MPI_Bcast(&BMatrixSize.RowCount,1,MPI_INT,0,MPI_COMM_WORLD);
		Time = MPI_Wtime();
		if (ParallelGetInvMatrix(A,AMatrixSize.ColCount,A,size))
		{
			// X = (A^-1)*B
			ParallelMulMatrix(A,AMatrixSize.RowCount,AMatrixSize.ColCount,B,BMatrixSize.RowCount,BMatrixSize.ColCount,X,size);
			Time = MPI_Wtime() - Time;
			printf("The system successfully solved. Time= %lf s\n",Time);
			printf("Writing X vector...\n");
			// запись вектора X
			if (fprintf(XFile,"%d %d\n",BMatrixSize.RowCount,1) > 0)
			{
				for (i = 0; i < BMatrixSize.RowCount; i++)
					if (fprintf(XFile,"%lf\n",X[i]) <= 0)
					{
						perror(argv[X_PARAM]);
						break;
					}
				if (i == BMatrixSize.RowCount) printf("%s successfully created\n",argv[X_PARAM]);
			}
			else perror(argv[X_PARAM]);
		}
		else
		{
			printf("Error: this method can not solve this system\n");
		}
		// освобождение памяти 
		free(A);
		free(B);
		free(X);
		// закрытие файлов
		fclose(AFile);
		fclose(BFile);
		fclose(XFile);
		// останавливаем ведомые
		Continue = 0;
		MPI_Bcast(&Continue,1,MPI_INT,0,MPI_COMM_WORLD);
	}
	else
		SlaveProc(rank,size);
	MPI_Finalize();
	return 0;
} 
