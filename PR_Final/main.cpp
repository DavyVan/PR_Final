#include<iostream>
#include<cstdlib>
#include<cmath>
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<stdio.h>
#include <tchar.h>
#include<Windows.h>
#include <strsafe.h>
#include<ctime>
#include<vector>
#include<process.h>
#pragma comment(lib, "User32.lib")
#include"hog.h"

#define WChar2Char(lpWideCharStr,cbWideChar,lpMultiByteStr,cbMultiByte) \
	WideCharToMultiByte(CP_ACP,0,lpWideCharStr,cbWideChar,lpMultiByteStr,cbMultiByte,NULL,NULL)

#define INPUT_NUM 1764//输入层节点数
//#define INPUT_NUM 36
#define HIDE_NUM 125//隐含层节点数
#define OUTPUT_NUM 1//输出层节点数

#define HOG_BLOCK_SIZE 2	//HOG中block的大小，每个方向包含多少个cell，正方形
#define HOG_BLOCK_STRIDE_SIZE 8		//HOG中block移动的步长，一般与cell大小相同
#define HOG_CELL_SIZE 8		//HOG中cell的大小，像素为单位，正方形
#define HOG_BIN_NUM 9		//HOG中梯度方向个数，即bin的个数
#define PI 3.14159265359
#define WIN_STEP 20//滑动窗口步长

//程序开关

//#define CALC_FEATURE	//是否执行特征向量计算步骤，计算的特征向量会保存在文件中，不能与下面两个阶段一趟进行
//#define TRAINING		//是否执行训练阶段，该阶段消耗大量时间，因为变量声明问题，不能CALC_FEATURE与TESTING阶段一趟进行
#define TESTING			//是否执行测试阶段，请确保完成执行前面两个阶段之后再打开此开关，且不能与上面两个阶段一趟进行
#define USE_MULTITHREAD	//是否开启多线程（CPU），用于更快的识别目标图片,5831ms/3223ms,Sp=1.8

using namespace std;
using namespace cv;

typedef struct//存储神经网络的结构体
{
	double alpha;//激活函数参数
	double beta;//学习率
	double weight1[HIDE_NUM][INPUT_NUM];//输入层到隐含层的权重
	double weight2[OUTPUT_NUM][HIDE_NUM];//隐含层到输出层的权重
}N_Network;

typedef struct//存储找到的Ground Truth矩形框
{
	int tl_r, tl_c; 
	int br_r, br_c;
	int height, width;
}Rect_Area;

typedef struct//存储分类器判别为行人的矩形框
{
	Mat_<uchar> img;
	Rect rect;
	double match_rate;
}Match_Rect;

typedef struct Node//存储特征向量的链表节点
{
	unsigned char num;
	int value;
	struct Node* next;
}LNode;

#ifdef USE_MULTITHREAD
typedef struct threadParams		//调用多线程时传参用的结构体
{
	int i;		//线程编号，用于跟踪处理进度
	int min_height;
	int min_width;
	int height_growth;
	int width_growth;
	int win_step;
	Mat_<uchar> *img_gray;
	N_Network *net;
	vector<Match_Rect> *match_list;
	HANDLE *match_list_mutex;
};
#endif


void LBP(Mat_<uchar> img, Mat_<uchar> &result, int x, int y)//计算一个像素点的LBP值
{
	unsigned char lbp=0;
	
	if(img(x-1, y-1)>img(x, y))
		lbp+=1;
	for(int i=0; i<2; i++)
	{
			lbp<<=1;
			if(img(x-1, y+i)>img(x, y))
				lbp+=1;
	}
	for(int i=0; i<2; i++)
	{
		lbp<<=1;
		if(img(x+i, y+1)>img(x, y))
			lbp+=1;
	}
	for(int i=0; i<2; i++)
	{
		lbp<<=1;
		if(img(x+i, y+1)>img(x, y))
			lbp+=1;
	}
	for(int i=0; i<2; i++)
	{
			lbp<<=1;
			if(img(x+1, y-i)>img(x, y))
				lbp+=1;
	}
	lbp<<=1;
	if(img(x, y-1)>img(x, y))
		lbp+=1;

	result(x, y)=lbp;
}

LNode* Get_Map(unsigned char *M)//得到初始LBP循环不变的映射表
{
	unsigned char min=255, num=0;
	int flag[256]={0};
	LNode *head=NULL, *cur=NULL, *pre=NULL, *temp=NULL;
	int count=0;

	for(int n=0; n<=255; n++)
	{
		num=(unsigned char)n;
		min=n;
		for(int m=1; m<8; m++)
		{
			num=(num<<7)|(num>>1);
			if(num<min)
				min=num;
		}

		M[n]=min;
		//cout<<n<<": "<<(int)min<<"\t";
		if(flag[min])
			continue;
		flag[min]=1;
		temp=(LNode*)malloc(sizeof(LNode));
		temp->num=min;
		temp->value=0;
		temp->next=NULL;
		count++;
		if(head==NULL)
			head=temp;
		else
		{
			cur=head;
			pre=NULL;
			while(cur!=NULL && min>cur->num)
			{
				pre=cur;
				cur=cur->next;
			}
			if(pre==NULL)
			{
				temp->next=head;
				head=temp;
			}
			else
			{
				pre->next=temp;
				temp->next=cur;
			}
		}
	}

	//cout<<count<<endl;
	return head;
}

void Get_Vector(Mat_<uchar> img, unsigned char *map, LNode *head)//生成LBP特征向量
{
	int vector[256]={0};
	for(int i=1; i<img.rows-1; i++)
		for(int j=1; j<img.cols-1; j++)
			vector[map[img(i, j)]]++;

	LNode *cur=head;
	while(cur!=NULL)
	{
		cur->value=0;
		cur=cur->next;
	}
	cur=head;
	while(cur!=NULL)
	{
		cur->value=vector[cur->num];
		cur=cur->next;
	}
}

void Calculate_LBP(Mat_<uchar> img, unsigned char map[], LNode *head, double vector[])//计算图像的LBP特征向量
{  
	Mat_<uchar> result;
	
	result.create(img.rows, img.cols);
	for(int i=1; i<img.rows-1; i++)
		for(int j=1; j<img.cols-1; j++)
			LBP(img, result, i, j);
			
	LNode *cur=NULL;
	int count=0;

	Get_Vector(result, map, head);
	cur=head;
	while(cur!=NULL)
	{
		vector[count]=(double)cur->value/(img.rows*img.cols);
		//cout<<vector[count]<<" ";
		count++;
		cur=cur->next;
	}
}

void Init_Network(N_Network *net)//初始化神经网络
{
	srand((int) time(0));
	net->alpha=0.8;
	net->beta=0.3;
	for(int j=0; j<HIDE_NUM; j++)
		for(int k=0; k<INPUT_NUM; k++)
			net->weight1[j][k]=(double)2*rand()/RAND_MAX-1;

	for(int i=0; i<OUTPUT_NUM; i++)
		for(int j=0; j<HIDE_NUM; j++)
			net->weight2[i][j]=(double)2*rand()/RAND_MAX-1;
}

void Predict(N_Network *net, double input[], double output[])//预测当前样例所属类
{
	double hide[HIDE_NUM]={0};
	
	//得到当前网络的输出
	for(int j=0; j<HIDE_NUM; j++)
	{
		for(int k=0; k<INPUT_NUM; k++)
			hide[j]=hide[j]+input[k]*net->weight1[j][k];
		hide[j]=1/(1+exp(-net->alpha*hide[j]));
	}

	for(int i=0; i<OUTPUT_NUM; i++)
	{
		for(int j=0; j<HIDE_NUM; j++)
			output[i]=output[i]+hide[j]*net->weight2[i][j];
		output[i]=1/(1+exp(-net->alpha*output[i]));
	}
}

double Train_Network(N_Network *net, double input[], double output[])//训练神经网络
{
	double hide[HIDE_NUM]={0}, pre_hide[HIDE_NUM]={0}, net_output[OUTPUT_NUM]={0}, pre_net_output[OUTPUT_NUM]={0};
	double error[OUTPUT_NUM]={0};
	double sum_error=0;

	//得到当前网络的输出
	for(int j=0; j<HIDE_NUM; j++)
	{
		for(int k=0; k<INPUT_NUM; k++)
			pre_hide[j]=pre_hide[j]+input[k]*net->weight1[j][k];
		hide[j]=1/(1+exp(-net->alpha*pre_hide[j]));
	}
	for(int i=0; i<OUTPUT_NUM; i++)
	{
		for(int j=0; j<HIDE_NUM; j++)
			pre_net_output[i]=pre_net_output[i]+hide[j]*net->weight2[i][j];
		net_output[i]=1/(1+exp(-net->alpha*pre_net_output[i]));
	}
	//计算期望输出与实际输出之间的误差
	for(int i=0; i<OUTPUT_NUM; i++)
	{
		error[i]=output[i]-net_output[i];
		sum_error+=pow(error[i], 2);
	}
	//修正网络中各边的权值
	for(int i=0; i<OUTPUT_NUM; i++)
	{
		double temp=net->beta*error[i]*(-net->alpha*exp(-net->alpha*pre_net_output[i])/pow(1+exp(-net->alpha*pre_net_output[i]), 2));

		for(int j=0; j<HIDE_NUM; j++)
		{
			for(int k=0; k<INPUT_NUM; k++)
			{
				net->weight1[j][k]=net->weight1[j][k]-temp*net->weight2[i][j]*net->weight2[i][j]*(-net->alpha*exp(-net->alpha*pre_hide[i])/pow(1+exp(-net->alpha*pre_hide[i]), 2))*input[k];
			}
			net->weight2[i][j]=net->weight2[i][j]-temp*hide[j];
		}
	}
	return sum_error;
}

void Copy_Image(Mat_<uchar> img, Rect rect, Mat_<uchar> &result)//将矩形区域提取为独立的图像
{
	for(int i=rect.tl().y; i<rect.br().y; i++)
		for(int j=rect.tl().x; j<rect.br().x; j++)
			result(i-rect.tl().y, j-rect.tl().x)=img(i, j);
}

void Find_Rectangle(Mat_<Vec3b> img, vector<Rect_Area> &rect_list)//寻找已经标记好的Ground Truth
{
	int sign=0;

	for(int i=0; i<img.rows; i++)
	{
		for(int j=0; j<img.cols; j++)
		{
			if(img(i, j)[2]==255 && img(i, j)[0]==0 && img(i, j)[1]==0)
			{
				sign=0;
				for(int k=0; k<rect_list.size(); k++)
				{
					if(i>=rect_list[k].tl_r && i<=rect_list[k].br_r && j>=rect_list[k].tl_c && j<=rect_list[k].br_c)
					{
						sign=1;
						break;
					}
				}
				if(sign==0)
				{
					int n=0;
					Rect_Area temp;

					while(img(i+n, j)[2]==255 && img(i+n, j)[0]==0 && img(i+n, j)[1]==0)
						n++;
					temp.tl_r=i;
					temp.br_r=i+n-1;
					n=0;
					while(img(i, j+n)[2]==255 && img(i,  j+n)[0]==0 && img(i,  j+n)[1]==0)
						n++;
					temp.tl_c=j;
					temp.br_c=j+n-1;
					temp.height=temp.br_r-temp.tl_r+1;
					temp.width=temp.br_c-temp.tl_c+1;
					rect_list.push_back(temp);
				}
			}
		}
	}
	
	for(int i=0; i<rect_list.size(); i++)
		cout<<"("<<rect_list[i].tl_r<<", "<<rect_list[i].tl_c<<") ("<<rect_list[i].tl_r<<", "<<rect_list[i].br_c<<") ("
		<<rect_list[i].br_r<<", "<<rect_list[i].tl_c<<") ("<<rect_list[i].br_r<<", "<<rect_list[i].br_c<<")"<<endl;
	
}
#define USE_CV_HOG

void Calculate_HOG(Mat_<uchar> img, double hog_vector[])//计算图像的HOG特征向量
{
#ifdef USE_CV_HOG
	HOGDescriptor *hog=new HOGDescriptor(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	vector<float> descriptors;//结果数组

	hog->compute(img, descriptors, Size(1,1), Size(0, 0)); //调用计算函数开始计算 
	//cout<<"HOG dims: "<<descriptors.size()<<endl;
	for(int i=0; i<descriptors.size(); i++)
		hog_vector[i]=descriptors[i];
#else
	Mat t_img = Flattening(img);
	int cell_cols = t_img.cols / CELL_SIZE[0] + 1;
	int cell_rows = t_img.rows / CELL_SIZE[1] + 1;

	vector<vector<vector<double>>> dest_hog_vec;

	InitVector(dest_hog_vec, cell_cols, cell_rows, GRADIENT_SIZE);
	CalcHOG(img, CELL_SIZE, BLOCK_SIZE, GRADIENT_SIZE, dest_hog_vec);

	int idx = 0;
	for (int j = 0; j < dest_hog_vec.size(); j++) {
		for (int i = 0; i < dest_hog_vec[j].size(); i++) {
			for (int k = 0; k < GRADIENT_SIZE; k++)
				hog_vector[idx++] = dest_hog_vec[i][j][k];
		}
	}
#endif
}

void Init_Black(Mat_<uchar> &img)//将图像初始化为黑色
{
	for(int i=0; i<img.rows; i++)
		for(int j=0; j<img.cols; j++)
			img(i, j)=0;
}

int HSV_Edge(Mat_<Vec3b> img, Mat_<uchar> &result)//计算彩色图像在HSV空间的边缘图像
{
	Mat_<Vec3b> temp(img.rows, img.cols);
	Mat_<uchar> edge[3];

	cvtColor(img, temp, CV_BGR2HSV);
	for(int i=0; i<temp.rows; i++)
		for(int j=0; j<temp.cols; j++)
			for(int k=0; k<3; k++)
			{
				edge[k].create(img.rows, img.cols);
				edge[k](i, j)=temp(i, j)[k];
			}
	for(int k=0; k<3; k++)
	{
		Laplacian(edge[k], edge[k], CV_8U);
		convertScaleAbs(edge[k], edge[k]);
	}

	int max=0;

	for(int i=3; i<temp.rows-3; i++)
		for(int j=3; j<temp.cols-3; j++)
		{
			result(i, j)=(unsigned char)(0.2*edge[0](i, j)+0.6*edge[1](i, j)+0.2*edge[2](i, j));//以不同的权重乘以HSI三个分量相加得到边缘图像
			if(result(i, j)>max)
				max=result(i, j);
		}
	return max;
}

double getMSSIM(const Mat& i1, const Mat& i2)//计算两张图片的MSSIM相似度
{ 
    const double C1=6.5025, C2=58.5225;
    int d=CV_32F;

    Mat I1, I2; 
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d); 

    Mat I2_2=I2.mul(I2);        // I2^2
    Mat I1_2=I1.mul(I1);        // I1^2
    Mat I1_I2=I1.mul(I2);        // I1 * I2

    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2=mu1.mul(mu1);    
    Mat mu2_2=mu2.mul(mu2); 
    Mat mu1_mu2=mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12; 

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2-=mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2-=mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12-=mu1_mu2;

    Mat t1, t2, t3; 

    t1=2*mu1_mu2+C1; 
    t2=2*sigma12+C2; 
    t3=t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1=mu1_2+mu2_2+C1; 
    t2=sigma1_2+sigma2_2+C2;     
    t1=t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim=mean( ssim_map ); // mssim = average of ssim map
    return mssim[0]; 
}

double Detect_Best(Mat_<uchar> img, vector<Mat_<uchar>> sample_list)//识别图片中的对象
{
	double max_rate=0;

	for(int i=0; i<sample_list.size(); i++)
	{
		double rate=getMSSIM(img, sample_list[i]);
		if(rate>max_rate)
			max_rate=rate;
	}
	return max_rate;
}

#ifdef USE_MULTITHREAD
unsigned int __stdcall Recognize_Multithread(PVOID params)
{
	threadParams *_params = (threadParams*) params;
	int i = _params->i;
	int min_height = _params->min_height;
	int min_width = _params->min_width;
	int height_growth = _params->height_growth;
	int width_growth = _params->width_growth;
	int win_step = _params->win_step;
	Mat_<uchar> *img_gray = _params->img_gray;
	N_Network *net = _params->net;
	vector<Match_Rect> *match_list = _params->match_list;
	HANDLE *match_list_mutex = _params->match_list_mutex;

	//cout << "Process: " << (double) i / 5 * 100 << "\%" << endl;
	int win_height = min_height + i*height_growth;
	int win_width = min_width + i*width_growth;
	for (int j = 0; (win_height + j*win_step)<img_gray->rows; j++)
	{
		for (int k = 0; (win_width + k*win_step)<img_gray->cols; k++)
		{
			Rect rect(k*win_step, j*win_step, win_width, win_height);
			Mat_<uchar> temp(win_height, win_width);
			//Mat_<Vec3b> temp(win_height, win_width);
			double input[INPUT_NUM] = { 0 }, output[OUTPUT_NUM] = { 0 };
			Match_Rect match;

			Copy_Image(*img_gray, rect, temp);
			//resize(temp, temp, Size(48, 96));
			//Calculate_LBP(temp, map, head, input);
			resize(temp, temp, Size(64, 64));
			Calculate_HOG(temp, input);
			Predict(net, input, output);
			if (output[0]>0.75)
			{
				match.img = temp.clone();
				match.rect = rect;
				match.match_rate = output[0];

				WaitForSingleObject(*match_list_mutex, INFINITE);
				match_list->push_back(match);
				ReleaseMutex(*match_list_mutex);
			}
		}
	}
	printf("Thread No.%d/6 completed!\n", i+1);
	return 0;
}
#endif

double pos_data[50000][INPUT_NUM], neg_data[50000][INPUT_NUM];

int main()
{
	

	clock_t start, finish;
	unsigned char map[256]={0};
	LNode *head=NULL;
	//double *pos_data[20000], *neg_data[20000];
	int pos_num=0, neg_num=0;
	
	head=Get_Map(map);

#ifdef CALC_FEATURE
	/////////////计算训练样本的特征向量//////////////
	FILE *pos_fp=fopen("C:\\Users\\FanQuan\\Desktop\\pos_hog_3.txt", "w");
	FILE *neg_fp=fopen("C:\\Users\\FanQuan\\Desktop\\neg_hog_3.txt", "w");
	
	WIN32_FIND_DATA ffd;
	HANDLE hFind=FindFirstFile(TEXT("C:\\Users\\FanQuan\\Desktop\\new_train\\*"), &ffd);
	char flag='0';
	char filename[1024], path[1024];
	//读取文件夹下的所有文件
	if (hFind != INVALID_HANDLE_VALUE)
	{
		//处理第一个找到的文件
		while(FindNextFile(hFind,&ffd))
		{
			flag='0';
			if(ffd.cFileName[0]=='n')//读到的图片为负例
				flag='0';
			else if(ffd.cFileName[0]=='p')//读到的图片为正例
				flag='1';
			else
				continue;
			WChar2Char(ffd.cFileName, -1, filename, 256);
			//cout<<filename<<endl;
			sprintf(path, "C:/Users/FanQuan/Desktop/new_train/%s", filename);
			//cout<<path<<" "<<endl;

			// 读入图片   
			Mat_<Vec3b> img_source=imread(path);
			Mat_<uchar> img(img_source.rows, img_source.cols);
	
			cvtColor(img_source, img, CV_BGR2GRAY);
			if(!img.data)
			{
				cout<<"Image read error!"<<endl;
				continue;
			}
			//resize(img, img, Size(48, 96));
			resize(img, img, Size(64, 64));

			char data[1000];
			if(flag=='0')
			{
				Calculate_LBP(img, map, head, neg_data[neg_num]);
				//Calculate_HOG(img, neg_data[neg_num]);
				for(int n=0; n<INPUT_NUM; n++)
				{
					sprintf(data, " %lf", neg_data[neg_num][n]);
					fputs(data, neg_fp);
				}
				fputc('\n', neg_fp);

				neg_num++;
			}
			else
			{
				Calculate_LBP(img, map, head, pos_data[pos_num]);
				//Calculate_HOG(img, pos_data[pos_num]);
				for(int n=0; n<INPUT_NUM; n++)
				{
					sprintf(data, " %lf", pos_data[pos_num][n]);
					fputs(data, pos_fp);
				}
				fputc('\n', pos_fp);

				pos_num++;
			}
		}
		FindClose(hFind);
	}
	fclose(pos_fp);
	fclose(neg_fp);
	cout<<pos_num<<"\t"<<neg_num<<endl;
#endif
	
#ifdef TRAINING
	///////////////////训练神经网络分类器/////////////////////
	FILE *pos_fp=fopen("C:\\Users\\FanQuan\\Desktop\\pos_hog_3.txt", "r");
	FILE *neg_fp=fopen("C:\\Users\\FanQuan\\Desktop\\neg_hog_3.txt", "r");
	char str[100000];
	
	while(!feof(pos_fp))
	{
		char *temp=NULL;

		fgets(str, 100000, pos_fp);
		if(strlen(str)<10)
			continue;
		temp=str;
		for(int n=0; n<INPUT_NUM; n++)
		{
			temp=strchr(temp, ' ');
			temp++;
			pos_data[pos_num][n]=strtod(temp, 0);
		}
		pos_num++;
	}
	fclose(pos_fp);

	while(!feof(neg_fp))
	{
		char *temp=NULL;

		fgets(str, 100000, neg_fp);
		if(strlen(str)<10)
			continue;
		temp=str;
		for(int n=0; n<INPUT_NUM; n++)
		{
			temp=strchr(temp, ' ');
			temp++;
			neg_data[neg_num][n]=strtod(temp, 0);
		}
		neg_num++;
	}
	fclose(neg_fp);
	cout<<"Positive Sample: "<<pos_num<<endl;
	cout<<"Negative Sample: "<<neg_num<<endl;

	N_Network *net=(N_Network*)malloc(sizeof(N_Network));
	int count=0;
	double sum_error=1;
	double output[OUTPUT_NUM]={0};
	int pos=0;

	//多次迭代输入训练数据，使误差小于规定值
	cout<<"开始训练"<<endl;
	start=clock();
	Init_Network(net);//初始化网络
	while(sum_error>1e-2 && count<150000)
	{
		sum_error=0;

		pos=rand()%pos_num;
		net->beta=0.1;
		output[0]=1;
		sum_error+=Train_Network(net, pos_data[pos], output);
		count++;

		pos=rand()%neg_num;
		net->beta=0.2;
		output[0]=0;
		sum_error+=Train_Network(net, neg_data[pos], output);
		count++;
	}
	finish=clock();
	cout<<"迭代次数: "<<count<<endl;
	cout<<"训练时间: "<<(double)(finish-start)/CLOCKS_PER_SEC<<"s"<<endl;
	
	FILE* net_fp=fopen("C:\\Users\\FanQuan\\Desktop\\network_hog_3.bin", "wb");//保存训练好的神经网络分类器
	fwrite(net, sizeof(N_Network), 1, net_fp);
	fclose(net_fp);
	
#endif

#ifdef TESTING
	////////////////////计算已标记图像中ground truth的坐标位置//////////////////////
	Mat_<Vec3b> img_source[3];
	vector<Rect_Area> rect_list[3];
	int min_height=-1, min_width=-1, max_height=-1, max_width=-1;
	int height_growth=0, width_growth=0;
	//已标记好ground truth的图像
	img_source[0]=imread("E:/学习相关/模式识别/模式识别大作业/PRtest_1.png");
	img_source[1]=imread("E:/学习相关/模式识别/模式识别大作业/PRtest_2.png");
	img_source[2]=imread("E:/学习相关/模式识别/模式识别大作业/PRtest_3.png");

	for(int i=0; i<3; i++)
	{
		if(!img_source[i].data)
		{
			cout<<"Image read error!"<<endl;
			system("pause");
			exit(-1);
		}
		cout<<"PRtest_"<<i<<endl;
		Find_Rectangle(img_source[i], rect_list[i]);//找到图像中所有ground truth
		
		for(int j=0; j<rect_list[i].size(); j++)
		{
			if(min_height==-1 || rect_list[i][j].height<min_height)
				min_height=rect_list[i][j].height;
			if(min_width==-1 || rect_list[i][j].width<min_width)
				min_width=rect_list[i][j].width;
			if(max_height==-1 || rect_list[i][j].height>max_height)
				max_height=rect_list[i][j].height;
			if(max_width==-1 || rect_list[i][j].width>max_width)
				max_width=rect_list[i][j].width;
		}
	}
	height_growth=ceil((double)(max_height-min_height)/5);
	width_growth=ceil((double)(max_width-min_width)/5);
	
	
	
	///////////////////////读取正样本作为相似度标准模板//////////////////////
	vector<Mat_<uchar>> sample_list;
	WIN32_FIND_DATA ffd;
	HANDLE hFind=FindFirstFile(TEXT("C:\\Users\\FanQuan\\Desktop\\pos_data\\*"), &ffd);
	char flag='0';
	char filename[1024], path[1024];

	//读取文件夹下的所有文件
	if (hFind != INVALID_HANDLE_VALUE)
	{
		//处理第一个找到的文件
		while(FindNextFile(hFind,&ffd))
		{
			WChar2Char(ffd.cFileName, -1, filename, 256);
			//cout<<filename<<endl;
			sprintf(path, "C:/Users/FanQuan/Desktop/pos_data/%s", filename);
			//cout<<path<<" "<<endl;
			if(!strcmp(filename, ".") || !strcmp(filename, ".."))
				continue;

			Mat_<Vec3b> img_source=imread(path);
			Mat_<uchar> img(img_source.rows, img_source.cols);
	
			cvtColor(img_source, img, CV_BGR2GRAY);
			if(!img.data)
			{
				cout<<"Image read error!"<<endl;
				continue;
			}
			//resize(img, img, Size(48, 96));
			resize(img, img, Size(64, 64));
			sample_list.push_back(img);
		}
		FindClose(hFind);
	}
	
	
	
	///////////////////////////使用分类器对目标图像进行目标图像识别////////////////////////
	N_Network *net=(N_Network*)malloc(sizeof(N_Network));
	FILE* net_fp=fopen("C:\\Users\\FanQuan\\Desktop\\network_hog.bin", "rb");//读入训练好的神经网络分类器

	fread(net, sizeof(N_Network), 1, net_fp);
	fclose(net_fp);

	Mat_<Vec3b> img_rgb=imread("E:/学习相关/模式识别/模式识别大作业/PR_origin_3.png");//待识别图像的地址
	Mat_<uchar> img_gray(img_rgb.rows, img_rgb.cols);
	int max_gray=0;
	vector<Match_Rect> match_list;

	cvtColor(img_rgb, img_gray, CV_BGR2GRAY);
	//Mat_<uchar> img_lap(img_rgb.rows, img_rgb.cols);

	//adaptiveThreshold(img_gray, img_lap, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 25, 10);//对灰度图做局部自适应二值化
	
	//Init_Black(img_lap);
	//max_gray=HSV_Edge(img_rgb, img_lap);//计算彩色图像在HSV空间的边缘图像，并获得最大灰度值
	//threshold(img_lap, img_lap, max_gray*0.04, 255, CV_THRESH_BINARY);//以图像最大灰度值的0.04作为阈值对边缘图像做全局二值化
	//namedWindow("Laplace");
	//imshow("Laplace", img_lap);
	
	int win_height=min_height, win_width=min_width;
	int win_step=WIN_STEP;
	//以不同尺度的窗口遍历整张图像，并将窗口中图像计算特征向量，然后送入分类器
#ifndef USE_MULTITHREAD
	start = clock();
	for(int i=0; i<=5; i++)
	{
		//cout<<"Process: "<<(double)i/5*100<<"\%"<<endl;
		win_height=min_height+i*height_growth;
		win_width=min_width+i*width_growth;
		for(int j=0; (win_height+j*win_step)<img_gray.rows; j++)
		{
			for(int k=0; (win_width+k*win_step)<img_gray.cols; k++)
			{
				Rect rect(k*win_step, j*win_step, win_width, win_height);
				Mat_<uchar> temp(win_height, win_width);
				//Mat_<Vec3b> temp(win_height, win_width);
				double input[INPUT_NUM]={0}, output[OUTPUT_NUM]={0};
				Match_Rect match;

				Copy_Image(img_gray, rect, temp);
				//resize(temp, temp, Size(48, 96));
				//Calculate_LBP(temp, map, head, input);
				resize(temp, temp, Size(64, 64));
				Calculate_HOG(temp, input);
				Predict(net, input, output);
				if(output[0]>0.75)
				{
					match.img=temp.clone();
					match.rect=rect;
					match.match_rate=output[0];
					match_list.push_back(match);
				}
			}
		}
		cout<<"Process: "<<(double)(i+1)/6*100<<"\%"<<endl;
	}
	finish = clock();
	cout << "串行识别时间：" << finish - start << "ms" << endl;
#else
	start = clock();

	threadParams params;
	params.min_height = min_height;
	params.min_width = min_width;
	params.height_growth = height_growth;
	params.width_growth = width_growth;
	params.win_step = win_step;
	params.img_gray = &img_gray;
	params.net = net;
	params.match_list = &match_list;
	HANDLE match_list_mutex = CreateMutex(NULL, FALSE, NULL);
	params.match_list_mutex = &match_list_mutex;

	HANDLE threads[6];
	threadParams _params[6];
	for (int i = 0; i <= 5; i++)
	{
		memcpy(&_params[i], &params, sizeof(threadParams));
		_params[i].i = i;
		threads[i] = (HANDLE) _beginthreadex(NULL, 0, Recognize_Multithread, (PVOID) &_params[i], 0, NULL);
	}

	WaitForMultipleObjects(6, threads, TRUE, INFINITE);
	CloseHandle(match_list_mutex);

	finish = clock();
	cout << "并行识别时间：" << finish - start << "ms" << endl;
#endif

	////////////////////////////筛选分类器识别的行人框中最佳的矩形框///////////////////////////
	Mat_<Vec3b> result=img_rgb.clone();
	vector<Match_Rect> reco_list;

	start=clock();
	for(int i=0; i<match_list.size(); i++)
	{
		double rate=Detect_Best(match_list[i].img, sample_list);
		if(rate>0.6)//判断矩形框中图像是否为最佳行人图像
		{
			int sign=0;
			//match_list[i].match_rate=rate;
			for(int j=0; j<reco_list.size(); j++)
			{
				Rect rect= match_list[i].rect & reco_list[j].rect;
				if((double)rect.area()/match_list[i].rect.area()>0.5 || (double)rect.area()/reco_list[j].rect.area()>0.5)//如果两个矩形框相交的面积比例超过一半，则保留匹配度较高的矩形框
				{
					reco_list[j].rect=match_list[i].match_rate>reco_list[j].match_rate? match_list[i].rect:reco_list[j].rect;
					sign=1;
					break;
				}
			}
			if(sign==0)
				reco_list.push_back(match_list[i]);
		}
	}
	finish=clock();
	cout<<"识别时间: "<<(double)(finish-start)/CLOCKS_PER_SEC<<"s"<<endl;
	for(int i=0; i<reco_list.size(); i++)
		rectangle(result, reco_list[i].rect, Scalar(0, 0, 255), 1);
		
	imwrite("E:/学习相关/模式识别/模式识别大作业/PR_result_1.png", result);
	namedWindow("Result");
	imshow("Result", result);
	waitKey(0);
#endif
	system("pause");

	return 0;
}
