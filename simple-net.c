/*
 * simple-net.c 
 * Simple Neural Network
 * Stephen Cook 
 * 10/12/2008
 *
 * Implement a back-propagation neural network with verious network configurations.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define FALSE 0
#define TRUE 1
#define LOW 0.00
#define HIGH 1.00
#define BIAS 1.00

#define MAX_DOUBLE +HUGE_VAL
#define MIN_DOUBLE -HUGE_VAL

#define sqr(x)	((x)*(x))
#define INPUT_FILENAME	"quadrant_data.csv"
#define TEST_DATA_FILE  "quadrant_test_data.csv"
#define NUM_INPUT_VECTORS 800
#define MAX_EPOCHS	1000
#define MIN_ERROR	0.0050

/* Network Configuration Settings */
#define NUM_LAYERS	5
#define N	2	/* Number of input nodes */
#define M	4	/* Number of output nodes */
//int	Nodes[NUM_LAYERS] = {N, 4, 25, 50, M};
//int	Nodes[NUM_LAYERS] = {N, 4, 3, M};
int	Nodes[NUM_LAYERS] = {N, 4, M};
#define ETA	0.189
#define ALPHA	0.1
#define GAIN	1.0


/* Structures */
typedef struct {
	int 		Nodes;
	double*		Output;
	double*		Error;
	double**	Weight;
	double**	WeightSave;
	double**	dWeight;
} LAYER;

typedef struct {
	LAYER**		Layer;
	LAYER*		InputLayer;
	LAYER*		OutputLayer;
	double		Alpha;
	double		Eta;
	double		Gain;
	double		Error;
	double		EpochError;
	double		TestEpochError;
} NET;


/* Global Variables */
int i;			// General Increments and record keeping
int incr[NUM_INPUT_VECTORS+1][2];
double tr[NUM_INPUT_VECTORS+1][3];
int tstincr[NUM_INPUT_VECTORS+1][2];   	// Test Increment Data
double tsttr[NUM_INPUT_VECTORS+1][3];	// Test Data
FILE *file;


double genRandom(double Low, double High) {
  return ((double) rand() / RAND_MAX) * (High-Low) + Low;
}


/* *************************** genNetwork - Initialize Network and Layers ******************************* */
void genNetwork(NET *Net) {
	int l;

	Net->Layer= (LAYER**) calloc(NUM_LAYERS+1, sizeof(LAYER*));

	for(l=0;l<NUM_LAYERS;l++) {
		Net->Layer[l] = (LAYER*) malloc(sizeof(LAYER));
    		Net->Layer[l]->Nodes      = Nodes[l];
    		Net->Layer[l]->Output     = (double*)  calloc(Nodes[l]+1, sizeof(double));
    		Net->Layer[l]->Error      = (double*)  calloc(Nodes[l]+1, sizeof(double));
    		Net->Layer[l]->Weight     = (double**) calloc(Nodes[l]+1, sizeof(double*));
    		Net->Layer[l]->WeightSave = (double**) calloc(Nodes[l]+1, sizeof(double*));
    		Net->Layer[l]->dWeight    = (double**) calloc(Nodes[l]+1, sizeof(double*));
    		Net->Layer[l]->Output[0]  = BIAS;
		if (l>0) {
      			for (i=1; i<=Nodes[l]; i++) {
       		 		Net->Layer[l]->Weight[i]     = (double*) calloc(Nodes[l-1]+2, sizeof(double));
       		 		Net->Layer[l]->WeightSave[i] = (double*) calloc(Nodes[l-1]+2, sizeof(double));
       		 		Net->Layer[l]->dWeight[i]    = (double*) calloc(Nodes[l-1]+2, sizeof(double));
      			}
    		}
  	}
  	Net->InputLayer  = Net->Layer[0];
  	Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
  	Net->Alpha       = ALPHA;
  	Net->Eta         = ETA;
  	Net->Gain        = GAIN;

}


void genRandomWeights(NET* Net) {
	int l,j;
	// Could change high/low to be based on heuristic in book - based on nodes, layers, etc.
  	for (l=1; l<NUM_LAYERS; l++) {
    		for (i=1; i<=Net->Layer[l]->Nodes; i++) {
      			for (j=0; j<=Net->Layer[l-1]->Nodes; j++) {
        			Net->Layer[l]->Weight[i][j] = genRandom(-0.5000, 0.5000); // low and high random numbers.
      			}
    		}
  	}
}



void setInput(NET* Net, double* Input) {
	//printf("Setting Input: \t");
  	for (i=1; i<=Net->InputLayer->Nodes; i++) {
    		Net->InputLayer->Output[i] = Input[i-1];
		//printf(" %lf ",Net->InputLayer->Output[i]);
  	}
	//printf("\n");
}

void getOutput(NET* Net, double* Output) {
  	for (i=1; i<=Net->OutputLayer->Nodes; i++) {
    		Output[i-1] = Net->OutputLayer->Output[i];
  	}
}





void layerPropagate(NET* Net, LAYER* Lower, LAYER* Upper) {
  	int  j,k;
  	double Sum;

  	for (k=1; k<=Upper->Nodes; k++) {
    		Sum = 0;
    		for (j=0; j<=Lower->Nodes; j++) {
      			Sum += Upper->Weight[k][j] * Lower->Output[j];
    		}
    		Upper->Output[k] = 1 / (1 + exp(-Net->Gain * Sum));
		//printf(" SUM(%d,%d)=%lf\tOutput[%d]=%lf \n",k,j,Sum,k,Upper->Output[k]);
  	}
}

void netPropagate(NET* Net) {
  	int l;

  	for (l=0; l<NUM_LAYERS-1; l++) {
		//printf("Propagating Layer %d\n",l);
    		layerPropagate(Net, Net->Layer[l], Net->Layer[l+1]);
  	}
}


void calcOutputError(NET* Net, double* Target) {
  	int  j;
  	double Out, Err;

  	Net->Error = 0;
  	for (j=1; j<=Net->OutputLayer->Nodes; j++) {
    		Out = Net->OutputLayer->Output[j];
    		Err = Target[j-1]-Out;
    		Net->OutputLayer->Error[j] = Net->Gain * Out * (1-Out) * Err;
    		Net->Error += 0.5 * sqr(Err);
  	}
}

void layerBackprop(NET* Net, LAYER* Upper, LAYER* Lower) {
  	int  k,j;
  	double Out, Err;

  	for (k=1; k<=Lower->Nodes; k++) {
    		Out = Lower->Output[k];
    		Err = 0;
    		for (j=1; j<=Upper->Nodes; j++) {
      			Err += Upper->Weight[j][k] * Upper->Error[j];
    		}
    		Lower->Error[k] = Net->Gain * Out * (1-Out) * Err;
  	}
}

void netBackprop(NET* Net) {
  	int l;
  	for (l=NUM_LAYERS-1; l>1; l--) {
    		layerBackprop(Net, Net->Layer[l], Net->Layer[l-1]);
  	}
}




void updateWeights(NET* Net) {
  	int  l,k,j;
  	double Out, Err, dWeight;

  	for (l=1; l<NUM_LAYERS; l++) {
    		for (k=1; k<=Net->Layer[l]->Nodes; k++) {
      			for (j=0; j<=Net->Layer[l-1]->Nodes; j++) {
        			Out = Net->Layer[l-1]->Output[j];
        			Err = Net->Layer[l]->Error[k];
        			dWeight = Net->Layer[l]->dWeight[k][j];
        			Net->Layer[l]->Weight[k][j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
        			Net->Layer[l]->dWeight[k][j] = Net->Eta * Err * Out;
      			}
    		}
  	}
}





void netRun(NET* Net, double* Input, double* Target, int Training) {
	double* Output;
	
	Output = (double*)  calloc(M+1, sizeof(double));

  	setInput(Net, Input);
  	netPropagate(Net);
  	getOutput(Net, Output);

  	calcOutputError(Net, Target);
	//printf("Target:{ %lf %lf %lf %lf } => Error: %lf\n",Target[0],Target[1],Target[2],Target[3],Net->Error);
  	if (Training) {
    		netBackprop(Net);
    		updateWeights(Net);
  	} 
}





void netTrain(NET* Net, int Epochs) {
	int j,n,l;
	//double Output[M];
	double Target[M];
	double tstTarget[M];
	
	for(l=0;l<Epochs;l++) {
		//printf("Epoch: %d\n",l);
		Net->EpochError=0.0;
		Net->TestEpochError=0.0;
		for (n=0; n<NUM_INPUT_VECTORS; n++) {
			for(j=0;j<M;j++) { 
				Target[j]=0.000;
				tstTarget[j]=0.000;
			}
			if(incr[n][0]==1) { Target[0]=1.00; } else if(incr[n][0]==2) { Target[1]=1.00; }
			else if(incr[n][0]==3) { Target[2]=1.00; } else if(incr[n][0]==4) { Target[3]=1.00; }
			
			//printf("%d %d  { %lf %lf %lf %lf }\n",n,incr[n][0],Target[0],Target[1],Target[2],Target[3]);
    			netRun(Net, tr[n], Target, TRUE);
    			Net->EpochError+=Net->Error;	
    			
    			if(tstincr[n][0]==1) { tstTarget[0]=1.00; } else if(tstincr[n][0]==2) { tstTarget[1]=1.00; }
			else if(tstincr[n][0]==3) { tstTarget[2]=1.00; } else if(tstincr[n][0]==4) { tstTarget[3]=1.00; }
			
    			netRun(Net, tsttr[n], tstTarget, FALSE);
    			Net->TestEpochError+=Net->Error;	
    			
    			
  		}
		//printf("\t EpochError: %f \t TestError: %lf\n",Net->EpochError,Net->TestEpochError);
	}

}


/* ********************************************* MAIN ************************************************ */
int main(int argc, char *argv[]) {

	char 	line[80];
	NET 	Net;
	int	Stop;
	int	randomAr[NUM_INPUT_VECTORS+1],last,temp,randomNum;
	int	numEpoch;
	double	lastTrainingError,deltaErr;


	int RandomizeTrainingSet=FALSE;

	srand(time(0));

	for(i=0;i<NUM_INPUT_VECTORS;i++) {
		randomAr[i]=i;
	}

/* Randomize the training data sets */
	if(RandomizeTrainingSet) {
		for(last=NUM_INPUT_VECTORS;last>1;last--) {
			randomNum=rand()%last;
			temp=randomAr[randomNum];	
			randomAr[randomNum]=randomAr[last - 1];
			randomAr[last - 1] = temp;
		}
	}

/* Read in the data from the file */
	file = fopen(INPUT_FILENAME,"r");
	for(i=0;i<NUM_INPUT_VECTORS;i++) {
		if(fgets(line, sizeof line, file) == NULL) {
			break;
                }
		temp=randomAr[i];
                sscanf(line,"%d,%d,%lf,%lf",&incr[temp][0],&incr[temp][1],&tr[temp][0],&tr[temp][1]);
                //sscanf(line,"%d,%d,%lf,%lf",&incr[i][0],&incr[i][1],&tr[i][0],&tr[i][1]);
        }

	fclose(file);

	file = fopen(TEST_DATA_FILE,"r");
	for(i=0;i<NUM_INPUT_VECTORS;i++) {
		if(fgets(line, sizeof line, file) == NULL) {
			break;
                }
		temp=randomAr[i];
                sscanf(line,"%d,%d,%lf,%lf",&tstincr[temp][0],&tstincr[temp][1],&tsttr[temp][0],&tsttr[temp][1]);
        }
	fclose(file);


	/*
	for(i=0;i<NUM_INPUT_VECTORS;i++) {
		printf("%d %d %lf %lf\n",tstincr[i][0],tstincr[i][1],tsttr[i][0],tsttr[i][1]);
	}
	*/

	genNetwork(&Net);   		// Create the network
	genRandomWeights(&Net);		// Fill it with random weights


	Stop = FALSE;
	numEpoch=0;
	lastTrainingError=100000.00;
/* Training Loop */
	do {
		netTrain(&Net, 1);
		deltaErr=fabs(Net.EpochError - lastTrainingError);
		if(deltaErr<MIN_ERROR || numEpoch>MAX_EPOCHS) Stop=TRUE;
		printf("%d, %lf, %lf\n",numEpoch,(double)Net.EpochError/(double)NUM_INPUT_VECTORS,(double)Net.TestEpochError/(double)NUM_INPUT_VECTORS);
		numEpoch++;
		lastTrainingError=Net.EpochError;	
		
	} while(!Stop);


	return(0);
}
