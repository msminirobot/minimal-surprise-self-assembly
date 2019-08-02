#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// header file for changing betw. different scenarios (MANL, ...)
#include "STD14.h"

int FIT_FUN; // fitness function
int MANIPULATION; // options: 'NONE', 'PRE' (Predefined), 'MAN' (Manipulation)
int EVOL; // evolution vs replay of genome

// 2D grid size
int SIZE_X;
int SIZE_Y;

// positions
struct pos {
    int x;
    int y;
};

// agent data
struct agent {
    int type; // type of sensor prediction (None, line, ...)
    struct pos coord;  // position coordinates
    struct pos heading; // heading vector
};

struct agent *p; //current position
struct agent *p_next; //next position

// ANN genomes
// current weights
float weight_actionNet[POP_SIZE][LAYERS][CONNECTIONS];
float weight_predictionNet[POP_SIZE][LAYERS][CONNECTIONS];
// hidden states prediction network
float hiddenBefore_predictionNet[NUM_AGENTS][HIDDENP];
// mutated weights
float newWeight_actionNet[POP_SIZE][LAYERS][CONNECTIONS];
float newWeight_predictionNet[POP_SIZE][LAYERS][CONNECTIONS];

// average predictions during run
float pred_return[SENSORS];

// predictions of agents
int predictions[NUM_AGENTS][SENSORS];

// action values
int current_action[NUM_AGENTS][MAX_TIME];

// numbering of evolutionary run
int COUNT = 0;


/* Function: activation
 * activation function (tanh)
 *
 * x: input value
 * returns: activation
 *
 */
float activation(float x) {
    return 2.0/(1.0+exp(-2.0*x))-1.0;
}

/* Function: propagate_actionNet
 * propagation of action network
 *
 * weight_action: weights of individual
 * input: input array [sensors, last action value]
 * returns: action output [action value, turn direction]
 *
 */
int *propagate_actionNet(float weight_action[LAYERS][CONNECTIONS], int input[INPUTA]) {
    float hidden_net[HIDDENA];
    float i_layer[INPUTA]; //inputs: x sensor values + 1 action value
    float net[OUTPUTA]; // output: action value + turn direction
    int i, j;
    int *action_output = malloc(OUTPUTA * sizeof(int));
    
    // calculate activation of input neurons
    // input value + threshold value / bias (= extra input of 1 and weight)
    for(i=0;i<INPUTA;i++){
        i_layer[i] = activation((float)input[i]*weight_action[0][2*i]
                                -weight_action[0][2*i+1]);
    }
    
    // hidden layer - 4 hidden neurons
    for(i=0;i<HIDDENA;i++) {
        hidden_net[i] = 0.0;
        
        //calculate input of hidden layer
        for(j=0;j<INPUTA;j++){
            hidden_net[i] += i_layer[j]*weight_action[1][INPUTA*i+j];
        }
    }
    // calculate input of output layer
    for(i=0; i<OUTPUTA; i++){ // outputs
        net[i] = 0.0;
        for(j=0;j<HIDDENA;j++){ // hidden layers
            net[i] += activation(hidden_net[j])*weight_action[2][HIDDENA*i+j];
        }
    }
    
    // map outputs to binary values
    // first output: action value
    if(activation(net[0]) < 0.0){
        action_output[0] = STRAIGHT;
    }
    else{
        action_output[0] = TURN;
    }
    
    // second output: turn direction
    if(activation(net[1]) < 0.0){
        action_output[1] = UP;
    }
    else{
        action_output[1] = DOWN;
    }
    
    return action_output;
}

/* Function: prediction_output
 *
 * value: output of prediction network
 * returns: 0 or 1
 *
 */
int prediction_output(float value){
    if(activation(value) < 0.0){
        return 0;
    }
    else{
        return 1;
    }
}

/*
 * Function: propagate_predictionNet
 * propagate prediction network
 *
 * weight_prediction: weights of individual
 * a: index of agent
 * input: input array
 *
 */
void propagate_predictionNet(float weight_prediction[LAYERS][CONNECTIONS], int a, int input[INPUTP]) {
    float hidden_net[HIDDENP];
    float i_layer[INPUTP];
    float net[OUTPUTP];
    int i, j;
    
    for(i=0;i<INPUTP;i++){ // outputs of input layer
        i_layer[i] = activation((float)input[i]*weight_prediction[0][2*i]
                                - weight_prediction[0][2*i+1]);
    }
    
    for(i=0;i<HIDDENP;i++) { // hidden layer neurons
        hidden_net[i] = 0.0;
        for(j=0;j<INPUTP;j++){ // inputs
            hidden_net[i] += i_layer[j]*weight_prediction[1][(INPUTP+1)*i+j]; // inputs: InputP + 1 recurrent
        }
        hidden_net[i] += hiddenBefore_predictionNet[a][i]*weight_prediction[1][(INPUTP+1)*i+INPUTP];
        hiddenBefore_predictionNet[a][i] = activation(hidden_net[i]);
    }
    
    for(i=0;i<OUTPUTP;i++) { //outputs (== # of sensors)
        net[i] = 0.0;
        for(j=0;j<HIDDENP;j++) { // hidden layers
            net[i] += activation(hidden_net[j])*weight_prediction[2][i*HIDDENP+j];
        }
    }
    
    if(MANIPULATION == NONE){
        // maps prediction values to binary sensor values
        for(i=0;i<OUTPUTP;i++) { // output values
            predictions[a][i] = prediction_output(net[i]);
        }
    }
    
    else if(MANIPULATION == MAN){ // prediction manipulation STD sensor model

        if(p[a].type == LINE){ // std sensor model (6 sensors)
            
            // predefined
            predictions[a][S0] = 1;
            predictions[a][S3] = 1;
            
            // learned
            predictions[a][S1] = prediction_output(net[0]);
            predictions[a][S2] = prediction_output(net[1]);
            
            predictions[a][S4] = prediction_output(net[2]);
            predictions[a][S5] = prediction_output(net[3]);
            
            if(SENSOR_MODEL == STDL){ // 14 sensors
                
                // predefined
                predictions[a][S8] = 1;
                predictions[a][S11] = 1;
                
                // learned
                predictions[a][S6] = prediction_output(net[4]);
                predictions[a][S7] = prediction_output(net[5]);
                
                predictions[a][S9] = prediction_output(net[6]);
                predictions[a][S10] = prediction_output(net[7]);
                
                predictions[a][S12] = prediction_output(net[8]);
                predictions[a][S13] = prediction_output(net[9]);
            }
        }
    }
}

/* Function: selectAndMutate(int maxID)
 * selects and mutates genomes for next generation
 *
 * maxID: population with maximum fitness
 * fitness: fitness of whole population (i.e. number of evaluated genomes)
 *
 */
void selectAndMutate(int maxID, float fitness[POP_SIZE]) {
    float sum1 = 0.0;
    float sum2 = 0.0;
    float pr[POP_SIZE], r;
    int i, j, k, ind;
    
    // total fitness - used for mutation
    for(ind=0;ind<POP_SIZE;ind++){
        sum1 += fitness[ind];
    }
    
    // relative fitness over individuals
    for(ind=0;ind<POP_SIZE;ind++) {
        sum2 += fitness[ind];
        pr[ind] = sum2/sum1;
    }
    
    for(ind=0;ind<POP_SIZE;ind++) {
        if(ind==maxID) { //population with maximum fitness - elitism of 1
            // keep weights as they are
            for(j=0;j<LAYERS;j++){
                for(k=0;k<CONNECTIONS;k++) {
                    newWeight_predictionNet[ind][j][k] = weight_predictionNet[maxID][j][k];
                    newWeight_actionNet[ind][j][k] = weight_actionNet[maxID][j][k];
                }
            }
        } else { // mutate weights of networks
            r = (float)rand()/(float)RAND_MAX;
            i = 0;
            while( (r > pr[i]) && (i<POP_SIZE-1) ){
                i++;
            }
            for(j=0;j<LAYERS;j++){
                for(k=0;k<CONNECTIONS;k++) {
                    newWeight_predictionNet[ind][j][k] = weight_predictionNet[i][j][k];
                    newWeight_actionNet[ind][j][k] = weight_actionNet[i][j][k];
                }
            }
            for(j=0;j<LAYERS;j++){
                for(k=0;k<CONNECTIONS;k++) {
                    // 0.1 == mutation operator 
                    if((float)rand()/(float)RAND_MAX < MUTATION){ // prediction network
                        newWeight_predictionNet[ind][j][k]
                        += 0.8 * (float)rand()/(float)RAND_MAX - 0.4;  // 0.8 * [0,1] - 0.4 --> [-0.4, 0.4]
                    }
                    if((float)rand()/(float)RAND_MAX < MUTATION){ // action network
                        newWeight_actionNet[ind][j][k]
                        += 0.8 * (float)rand()/(float)RAND_MAX - 0.4;
                    }
                }
            }
        }
    }
    
    // update weights
    for(ind=0;ind<POP_SIZE;ind++) {
        for(j=0;j<LAYERS;j++){
            for(k=0;k<CONNECTIONS;k++) {
                weight_predictionNet[ind][j][k] = newWeight_predictionNet[ind][j][k];
                weight_actionNet[ind][j][k] = newWeight_actionNet[ind][j][k];
            }
        }
    }
}

/* Function: adjustXPosition
 * adjusts position for movement on torus
 *
 * x: x position
 * return: updated x value
 *
 */
int adjustXPosition(int x){
    if(x < 0){
        x += SIZE_X;
    }
    else if(x > SIZE_X-1){
        x -= SIZE_X;
    }
    
    return x;
}

/* Function: adjustYPosition
 * adjusts position for movement on torus
 *
 * y: y position
 * return: updated y value
 *
 */
int adjustYPosition(int y){
    if(y < 0){
        y += SIZE_Y;
    }
    else if(y > SIZE_Y-1){
        y -= SIZE_Y;
    }
    
    return y;
}


/* Function: sensorModelSTD
 * sensor model with 6 binary values in heading direction
 *
 * i: index of current agent
 * grid: array with agents positions (0 - no agent, 1 - agent)
 * returns: sensor values
 *
 */
int * sensorModelSTD(int i, int grid[SIZE_X][SIZE_Y]){
    int j;
    int dy, dx, dxl, dyl, dxplus, dxmin, dyplus, dymin;
    int *sensors = malloc(SENSORS * sizeof(int));
    
    // initialise values / reset sensor values
    for(j=0; j<SENSORS; j++){
        sensors[j] = 0;
    }
    
    // determine coordinates of cells looked at
    
    // short range forward
    dx = adjustXPosition(p[i].coord.x + p[i].heading.x);
    dy = adjustYPosition(p[i].coord.y + p[i].heading.y);
    
    // long range forward
    dxl = adjustXPosition(p[i].coord.x + 2*p[i].heading.x);
    dyl = adjustYPosition(p[i].coord.y + 2*p[i].heading.y);
    
    // points for left and right sensors
    dyplus = adjustYPosition(p[i].coord.y + 1); // y coordinate + 1
    dymin = adjustYPosition(p[i].coord.y - 1); // y coordinate - 1
    
    dxplus = adjustXPosition(p[i].coord.x + 1); // x coordinate + 1
    dxmin = adjustXPosition(p[i].coord.x - 1); // x coordinate - 1
    
    // forward looking sensor / in direction of heading
    sensors[S0] = grid[dx][dy]; // FORWARD SHORT
    sensors[S3] = grid[dxl][dyl]; // FORWARD LONG

    // headings in x-direction (i.e., y equals 0)
    if(p[i].heading.y == 0){
        if(p[i].heading.x == 1){
            sensors[S2] = grid[dx][dyplus]; // y+1; x + 1 * heading
            sensors[S5] = grid[dxl][dyplus]; // y+1; x + 2 * heading
            sensors[S1] = grid[dx][dymin]; // y-1; x + 1 * heading
            sensors[S4] = grid[dxl][dymin]; // y-1; x + 2 * heading
        }
        else{
            sensors[S1] = grid[dx][dyplus]; // y+1; x + 1 * heading
            sensors[S4] = grid[dxl][dyplus]; // y+1; x + 2 * heading
            sensors[S2] = grid[dx][dymin]; // y-1; x + 1 * heading
            sensors[S5] = grid[dxl][dymin]; // y-1; x + 2 * heading
        }
    }
    
    // headings in y-direction (i.e., x equals 0)
    else if(p[i].heading.x == 0){
        if(p[i].heading.y == 1){
            sensors[S1] = grid[dxplus][dy]; // y + 1 * heading; x + 1
            sensors[S4] = grid[dxplus][dyl]; // y + 2 * heading; x + 1
            sensors[S2] = grid[dxmin][dy]; // y + 1 * heading; x - 1
            sensors[S5] = grid[dxmin][dyl]; // y + 2 * heading; x - 1
        }
        else{
            sensors[S2] = grid[dxplus][dy]; // y + 1 * heading; x + 1
            sensors[S5] = grid[dxplus][dyl]; // y + 2 * heading; x + 1
            sensors[S1] = grid[dxmin][dy]; // y + 1 * heading; x - 1
            sensors[S4] = grid[dxmin][dyl]; // y + 2 * heading; x - 1
        }
    }
    
    return sensors;
}

/* Function: sensorModelSTDL
 * sensor model with 14 sensors surrounding the agent
 *
 * i: index of current agent
 * grid: array with occupied grid cells (0 - no agent, 1 - agent)
 * returns: sensor values
 */
int * sensorModelSTDL(int i, int grid[SIZE_X][SIZE_Y]){
    int j;
    int dy, dx, dxl, dyl, dxplus, dxmin, dyplus, dymin, dxb, dyb, dxbl, dybl;
    int *sensors = malloc(SENSORS * sizeof(int));
    
    // initialise values / reset sensor values
    for(j=0; j<SENSORS; j++){
        sensors[j] = 0;
    }
    
    // determine coordinates of cells looked at
    
    // short range forward
    dx = adjustXPosition(p[i].coord.x + p[i].heading.x); // x coordinate + heading
    dy = adjustYPosition(p[i].coord.y + p[i].heading.y); // y coordinate + heading
    
    // long range forward
    dxl = adjustXPosition(p[i].coord.x + 2*p[i].heading.x); // x coordinate + 2 * heading
    dyl = adjustYPosition(p[i].coord.y + 2*p[i].heading.y); // y coordinate + 2 * heading
    
    // points for left and right sensors
    dyplus = adjustYPosition(p[i].coord.y + 1); // y coordinate + 1
    dymin = adjustYPosition(p[i].coord.y - 1); // y coordinate - 1
    
    dxplus = adjustXPosition(p[i].coord.x + 1); // x coordinate + 1
    dxmin = adjustXPosition(p[i].coord.x - 1); // x coordinate - 1
    
    // short range backwards
    dxb = adjustXPosition(p[i].coord.x - p[i].heading.x); // x coordinate - heading (backwards)
    dyb = adjustYPosition(p[i].coord.y - p[i].heading.y); // y coordinate - heading (backwards)
    
    // long range backwards
    dxbl = adjustXPosition(p[i].coord.x - 2*p[i].heading.x); // x coordinate - 2 * heading
    dybl = adjustYPosition(p[i].coord.y - 2*p[i].heading.y); // y coordinate - 2 * heading
    
    // forward looking sensor / in direction of heading
   
    sensors[S0] = grid[dx][dy]; // FORWARD SHORT
    sensors[S3] = grid[dxl][dyl]; // FORWARD LONG
    sensors[S8] = grid[dxb][dyb]; // backward short
    sensors[S11] = grid[dxbl][dybl]; // backward long
    
    if(p[i].heading.y == 0){ // headings in x-direction (i.e., y equals 0)
        if(p[i].heading.x == 1){
            
            sensors[S2] = grid[dx][dyplus]; // y + 1 and x + 1 * heading
            sensors[S1] = grid[dx][dymin]; // y - 1 and x + 1 * heading
            sensors[S5] = grid[dxl][dyplus]; // y + 1 and x + 2 * heading
            sensors[S4] = grid[dxl][dymin]; // y - 1 and x + 2 * heading
            sensors[S7] = grid[p[i].coord.x][dyplus]; // left/right of agent
            sensors[S6] = grid[p[i].coord.x][dymin]; // left/right of agent
            sensors[S10] = grid[dxb][dyplus]; // y + 1 and x - 1 * heading
            sensors[S9] = grid[dxb][dymin]; // y - 1 and x - 1 * heading
            sensors[S13] = grid[dxbl][dyplus]; // y + 1 and x - 2 * heading
            sensors[S12] = grid[dxbl][dymin]; // y - 1 and x - 2 * heading

        }
        else{ // heading.x == -1
            // forward left / right
            sensors[S1] = grid[dx][dyplus]; // y + 1 and x + 1 * heading
            sensors[S2] = grid[dx][dymin]; // y - 1 and x + 1 * heading
            sensors[S4] = grid[dxl][dyplus]; // y + 1 and x + 2 * heading
            sensors[S5] = grid[dxl][dymin]; // y - 1 and x + 2 * heading
            // left / right of agent
            sensors[S6] = grid[p[i].coord.x][dyplus];
            sensors[S7] = grid[p[i].coord.x][dymin];
            // backward left / right
            sensors[S9] = grid[dxb][dyplus]; // y + 1 and x - 1 * heading
            sensors[S10] = grid[dxb][dymin]; // y - 1 and x - 1 * heading
            sensors[S12] = grid[dxbl][dyplus]; // y + 1 and x - 2 * heading
            sensors[S13] = grid[dxbl][dymin]; // y - 1 and x - 2 * heading
        }
    }
    else if(p[i].heading.x == 0){ // headings in y-direction (i.e., x equals 0)
        if(p[i].heading.y == 1){
            // forward left / right
            sensors[S1] = grid[dxplus][dy]; // y + 1 * heading; x + 1
            sensors[S4] = grid[dxplus][dyl]; // y + 2 * heading; x + 1
            sensors[S2] = grid[dxmin][dy]; // y + 1 * heading; x - 1
            sensors[S5] = grid[dxmin][dyl]; // y + 2 * heading; x - 1
            // left / right of agent
            sensors[S6] = grid[dxplus][p[i].coord.y];
            sensors[S7] = grid[dxmin][p[i].coord.y];
            // backwards left / right
            sensors[S10] = grid[dxmin][dyb]; // y - 1 * heading, x - 1
            sensors[S9] = grid[dxplus][dyb]; // y - 1 * heading, x + 1
            sensors[S13] = grid[dxmin][dybl]; // y - 2 * heading, x - 1
            sensors[S12] = grid[dxplus][dybl]; // y - 2 * heading, x + 1
        }
        else{ // heading.y == -1
            // forward left / right
            sensors[S2] = grid[dxplus][dy]; // y + 1 * heading; x + 1
            sensors[S5] = grid[dxplus][dyl]; // y + 2 * heading; x + 1
            sensors[S1] = grid[dxmin][dy]; // y + 1 * heading; x - 1
            sensors[S4] = grid[dxmin][dyl]; // y + 2 * heading; x - 1
            // left / right of agent
            sensors[S7] = grid[dxplus][p[i].coord.y];
            sensors[S6] = grid[dxmin][p[i].coord.y];
            // backwards left / right
            sensors[S9] = grid[dxmin][dyb]; // y - 1 * heading, x - 1
            sensors[S10] = grid[dxplus][dyb]; // y - 1 * heading, x + 1
            sensors[S12] = grid[dxmin][dybl]; // y - 2 * heading, x - 1
            sensors[S13] = grid[dxplus][dybl]; // y - 2 * heading, x + 1
        }
    }
    
    return sensors;
}


/* Function: sensorModelSTDSL
 * sensor model with 8 sensors covering Moore neighborhood
 *
 * i: index of current agent
 * grid: array with occupied grid cells (0 - no agent, 1 - agent)
 * returns: sensor values
 *
 */
int * sensorModelSTDSL(int i, int grid[SIZE_X][SIZE_Y]){
    int j;
    int dy, dx, dxplus, dxmin, dyplus, dymin, dxb, dyb;
    int *sensors = malloc(SENSORS * sizeof(int));
    
    // initialise values / reset sensor values
    for(j=0; j<SENSORS; j++){
        sensors[j] = 0;
    }
    
    // determine coordinates of cells looked at

    // short range forward
    dx = adjustXPosition(p[i].coord.x + p[i].heading.x); // x coordinate + heading
    dy = adjustYPosition(p[i].coord.y + p[i].heading.y); // y coordinate + heading
    
    // points for left and right sensors
    dyplus = adjustYPosition(p[i].coord.y + 1); // y coordinate + 1
    dymin = adjustYPosition(p[i].coord.y - 1); // y coordinate - 1
    
    dxplus = adjustXPosition(p[i].coord.x + 1); // x coordinate + 1
    dxmin = adjustXPosition(p[i].coord.x - 1); // x coordinate - 1
    
    // short range backwards
    dxb = adjustXPosition(p[i].coord.x - p[i].heading.x); // x coordinate - heading (backwards)
    dyb = adjustYPosition(p[i].coord.y - p[i].heading.y); // y coordinate - heading (backwards)
    
    // forward looking sensor / in direction of heading
    sensors[S0] = grid[dx][dy]; // FORWARD SHORT
    sensors[S5] = grid[dxb][dyb]; // backward short
    
    if(p[i].heading.y == 0){ // headings in x-direction (i.e., y equals 0)
        if(p[i].heading.x == 1){
            
            sensors[S2] = grid[dx][dyplus]; // y + 1 and x + 1 * heading
            sensors[S1] = grid[dx][dymin]; // y - 1 and x + 1 * heading
            sensors[S4] = grid[p[i].coord.x][dyplus]; // left/right of agent
            sensors[S3] = grid[p[i].coord.x][dymin]; // left/right of agent
            sensors[S7] = grid[dxb][dyplus]; // y + 1 and x - 1 * heading
            sensors[S6] = grid[dxb][dymin]; // y - 1 and x - 1 * heading
        }
        else{ // heading.x == -1
            // forward left / right
            sensors[S1] = grid[dx][dyplus]; // y + 1 and x + 1 * heading
            sensors[S2] = grid[dx][dymin]; // y - 1 and x + 1 * heading
            // left / right of agent
            sensors[S3] = grid[p[i].coord.x][dyplus];
            sensors[S4] = grid[p[i].coord.x][dymin];
            // backward left / right
            sensors[S6] = grid[dxb][dyplus]; // y + 1 and x - 1 * heading
            sensors[S7] = grid[dxb][dymin]; // y - 1 and x - 1 * heading
        }
    }
    else if(p[i].heading.x == 0){ // headings in y-direction (i.e., x equals 0)
        if(p[i].heading.y == 1){
            // forward left / right
            sensors[S1] = grid[dxplus][dy]; // y + 1 * heading; x + 1
            sensors[S2] = grid[dxmin][dy]; // y + 1 * heading; x - 1
            // left / right of agent
            sensors[S3] = grid[dxplus][p[i].coord.y];
            sensors[S4] = grid[dxmin][p[i].coord.y];
            // backwards left / right
            sensors[S7] = grid[dxmin][dyb]; // y - 1 * heading, x - 1
            sensors[S6] = grid[dxplus][dyb]; // y - 1 * heading, x + 1
        }
        else{ // heading.y == -1
            // forward left / right
            sensors[S2] = grid[dxplus][dy]; // y + 1 * heading; x + 1
            sensors[S1] = grid[dxmin][dy]; // y + 1 * heading; x - 1
            // left / right of agent
            sensors[S4] = grid[dxplus][p[i].coord.y];
            sensors[S3] = grid[dxmin][p[i].coord.y];
            // backwards left / right
            sensors[S6] = grid[dxmin][dyb]; // y - 1 * heading, x - 1
            sensors[S7] = grid[dxplus][dyb]; // y - 1 * heading, x + 1
        }
    }
    
    return sensors;
}

/*
 * Function: doRun
 * contains loop for evaluation of one individual (genome)
 *
 * gen: generation of evolutionary process
 * ind: index of individual
 * p_initial: array with initial agent positions
 * maxTime: run time in time steps
 * log: logging of agent trajectory
 * noagents: number of agents which are currently used (use for self-repair/replay scenario only! otherwise to be set to NUM_AGENTS)
 * returns: fitness of run
 *
 */
float doRun(int gen, int ind, struct agent p_initial[], int maxTime, int log, int noagents){
    FILE *f;
    int i, j; // for loops
    int timeStep = 0; //current time step
    int fit = 0;
    float fit_return = 0.0;
    struct agent *temp;
    int occupied = 0;
    int predReturn[SENSORS] = { 0 }; // average predecitions
    double angle;
    struct pos tmp_agent_next;
    int inputA[INPUTA]; // input array action network
    int inputP[INPUTP]; // input array prediction network
    int *sensors = NULL;
    int *action_output = NULL;
    int grid[SIZE_X][SIZE_Y];
    
    // file name agent trajectory
    char str[12];
    char trajectory_file[100];
    int no_agents = noagents; // for replay runs or rather self-repair

    sprintf(str, "%d_%d_%d", COUNT, 0, no_agents);
    strcpy(trajectory_file, "agent_trajectory");
    strcat(trajectory_file, str);
    
    if(log==1){
        f = fopen(trajectory_file, "a");
        fprintf(f, "Gen: %d\n", gen);
        fprintf(f, "Grid: %d, %d\n", SIZE_X, SIZE_Y);
        fprintf(f, "Agents: %d\n", no_agents);
        fclose(f);
    }
    
    // initialise hidden neurons and predictions, prediction counter to zero
    memset(pred_return, 0, sizeof(pred_return));
    memset(predictions, 0, sizeof(predictions));
    memset(hiddenBefore_predictionNet, 0, sizeof(hiddenBefore_predictionNet));
    
    //initialise agents
    for(i=0; i<no_agents; i++){
        // set initial agent positions
        p[i].coord.x = p_initial[i].coord.x;
        p[i].coord.y = p_initial[i].coord.y;
        p[i].heading.x = p_initial[i].heading.x;
        p[i].heading.y = p_initial[i].heading.y;
        
        // next position
        p_next[i].coord.x = p_initial[i].coord.x;
        p_next[i].coord.y = p_initial[i].coord.y;
        p_next[i].heading.x = p_initial[i].heading.x;
        p_next[i].heading.y = p_initial[i].heading.y;
        
        // set manipulation initialisation based agent type
        if(MANIPULATION != NONE){
            if(p[i].type == LINE){
                if(SENSOR_MODEL == STDSL){
                    predictions[i][S0] = 1;
                    predictions[i][S5] = 1;
                }
                else{
                    predictions[i][S0] = 1;
                    predictions[i][S3] = 1;
                
                    if(SENSOR_MODEL == STDL){
                        predictions[i][S8] = 1;
                        predictions[i][S11] = 1;
                    }
                }
            }
            else if(p[i].type == SQUARE){ //
                if(SENSOR_MODEL == STDL){
                    predictions[i][S3] = 1;
                    predictions[i][S11] = 1;
                }
            }
            else if(p[i].type == DIAMOND){
                if(SENSOR_MODEL == STDL){
                    predictions[i][S1] = 1;
                    predictions[i][S2] = 1;
                    predictions[i][S3] = 1;
                    predictions[i][S9] = 1;
                    predictions[i][S10] = 1;
                    predictions[i][S11] = 1;
                }
            }
            else if(p[i].type == PAIR){
                predictions[i][S0] = 1;
            }
        }
    }
    
    while(timeStep < maxTime){
        
        // determine occupied grid cells (0 - unoccupied, 1 - occupied)
        memset(grid, 0, sizeof(grid));
        for(i=0; i<no_agents; i++){
            grid[p[i].coord.x][p[i].coord.y] = 1;
        }
        
        // iterate through all agents
        for(i=0;i<no_agents;i++){
            
            // store agent trajectory
            if(log == 1){
                f = fopen(trajectory_file, "a");
                // print position and heading
                fprintf(f, "%d: %d, %d, %d, %d\n", timeStep, p[i].coord.x, p[i].coord.y, p[i].heading.x, p[i].heading.y);
                fclose(f);
            }
            
            /*
             * Determine current sensor values (S of t)
             */
            free(sensors);
            sensors = NULL;
            
            if(SENSOR_MODEL == STDS){
                sensors = sensorModelSTD(i, grid);
            }
            else if(SENSOR_MODEL == STDL){
                sensors = sensorModelSTDL(i, grid);
            }
            else if(SENSOR_MODEL == STDSL){
                sensors = sensorModelSTDSL(i, grid);
            }
            
            for(j=0;j<SENSORS;j++){
                
                // set sensor values as ANN input values
                inputA[j] = sensors[j];
                inputP[j] = sensors[j];
                
                // count correct predictions - only evolving agents are included in fitness calculation
                if(sensors[j] == predictions[i][j]){
                    if(FIT_FUN == PRED){ // calculate fitness over course of time
                        fit++;
                    }
                }

                // set prediction counters
                predReturn[j] += predictions[i][j];

            } // for loop sensor values
            
            /*
             * propagate action network with current sensor values
             * and last action - output: next action [0, 1]
             */
        
            if(timeStep <= 0){ // initial action value
                inputA[SENSORS] = STRAIGHT; // 0 - SENSORS-1 = sensor values; SENSORS = action value
            } else {
                inputA[SENSORS] = current_action[i][timeStep-1]; // last action
            }
            
            free(action_output);
            action_output = NULL;
            
            action_output = propagate_actionNet(weight_actionNet[ind], inputA);

            // store actions over course of time per agent
            current_action[i][timeStep] = action_output[0];
            
            /*
             * propagate prediction network with current sensor
             * values and next action (action returned
             * by action ANN) to determine next sensor values
             * output: prediction of sensor values (per agent)
             */
            inputP[SENSORS] = current_action[i][timeStep]; // current action
            
            // prediction network only necessary when predictions aren't completely predefined
            if(MANIPULATION != PRE){
                propagate_predictionNet(weight_predictionNet[ind], i, inputP);
            }
            
            /* Update position / heading according to action value returned by Action ANN */
            
            p_next[i] = p[i];  // copy values
            
            // 0 == move straight; 1 == turn
            if(current_action[i][timeStep] == STRAIGHT){ // move 1 grid cell straight
                occupied = 0;
                
                // movement only possible when cell in front is not occupied (sensor S0)
                if(sensors[S0] == 0){
                    // move in heading direction (i.e. straight)
                    tmp_agent_next.x = adjustXPosition(p[i].coord.x + p[i].heading.x);
                    tmp_agent_next.y = adjustYPosition(p[i].coord.y + p[i].heading.y);
                    
                    // check if next cell is already occupied by agent
                    // next agent positions as far as updated (otherwise positions already checked via sensors)
                    for(int k=0; k<i; k++){
                        if(p_next[k].coord.x == tmp_agent_next.x && p_next[k].coord.y == tmp_agent_next.y){
                            occupied = 1;
                            break;
                        }
                    }
                    
                    if(!occupied){ // cell not occupied - agent can move
                        p_next[i].coord.x = tmp_agent_next.x;
                        p_next[i].coord.y = tmp_agent_next.y;
                    }
                }
            }
            else if (current_action[i][timeStep] == TURN) {  // turn - update heading
                // turn direction = action output 1
                // [-1, 1] --> defines turn direction
                
                angle = atan2(p[i].heading.y, p[i].heading.x); // calculate current orientation
                p_next[i].heading.x = cos(angle + action_output[1]*(PI/2));
                p_next[i].heading.y = sin(angle + action_output[1]*(PI/2));
            }
        } // agents
        
        timeStep++; //increase time step
        
        // update agent positions to next positions
        temp = p;
        p = p_next;
        p_next = temp;
        
    } // while time
    
    // prediction counter
    for(i=0; i<SENSORS; i++){
        pred_return[i] = (float)predReturn[i]/(float)(maxTime*no_agents);
    }
    
    if(log==1){ // print last time step
        f = fopen(trajectory_file, "a");
        for(i=0; i<no_agents; i++){
            fprintf(f, "%d: %d, %d, %d, %d\n", maxTime, p[i].coord.x, p[i].coord.y, p[i].heading.x, p[i].heading.y);
        }
        fclose(f);
    }
    
    if(FIT_FUN == PRED){ // normalize by number of agents which evolve (which is NUM_Agents - no)
        fit_return = (float)fit/(float)(no_agents * maxTime * SENSORS);
    }
    
    // return average fitness
    return fit_return;
}

/* Function: evolution
 *
 * no_agents: evolving agents
 *
 */
void evolution(int no_agents){
    // variables
    int i, b, j, ind, k, gen, rep;
    int maxID = -1;
    float max, avg;
    struct agent p_initial[REPETITIONS][NUM_AGENTS];
    float tmp_fitness;
    int max_rep = 0;
    int store = 0; //bool to store current agent and block data
    float agentPrediction[SENSORS];
    float fitness[POP_SIZE]; // store fitness of whole population for roulette wheel selection
    float pred[SENSORS]; // average predictions of best individual
    int action_values[NUM_AGENTS][MAX_TIME]; // action values of stored run
    int grid[SIZE_X][SIZE_Y];
    FILE *f;

    // store agent movement
    struct agent agent_maxfit[NUM_AGENTS];
    struct agent tmp_agent_maxfit_final[NUM_AGENTS];
    struct agent agent_maxfit_beginning[NUM_AGENTS];
    int tmp_action[NUM_AGENTS][MAX_TIME]; // tmp storage action values
    
    // file names
    char str[12];
    char predGen_file[100];
    char fit_file[100];
    char actVal_file[100];
    char agent_file[100];
    char actGen_file[100];
    
    // file names
    sprintf(str, "%d_%d_%d", COUNT, 0, no_agents);
    strcpy(fit_file, "fitness");
    strcat(fit_file, str);
    
    strcpy(predGen_file, "prediction_genomes");
    strcat(predGen_file, str);
    
    strcpy(actGen_file, "action_genomes");
    strcat(actGen_file, str);
    
    strcpy(actVal_file, "actionValues");
    strcat(actVal_file, str);
    
    strcpy(agent_file, "agents");
    strcat(agent_file, str);
    
    // initialise weights of neural nets in range [-0.5, 0.5]
    for(ind=0; ind<POP_SIZE; ind++){
        for(j=0; j<LAYERS; j++){
            for(k=0; k<CONNECTIONS; k++){
                weight_actionNet[ind][j][k] = 1.0 * (float)rand()/(float)RAND_MAX - 0.5;
                weight_predictionNet[ind][j][k] = 1.0 * (float)rand()/(float)RAND_MAX - 0.5;
            }
        }
    }
    
    // evolutionary runs
    for(gen=0;gen<MAX_GENS;gen++){ // generation level (iterate through populations)
        
        max = 0.0; // max. fitness of generation
        avg = 0.0; // average fitness of generation
        maxID = -1; // id of individual with max. fitness
        
        // initialisation of starting positions (all genomes have same set of starting positions)
        for(k=0; k<REPETITIONS; k++){
            
            memset(grid, 0, sizeof(grid));
    
            // generate agent positions
            for(i=0;i<NUM_AGENTS;i++){
                // initialise agent positions to random discrete x & y values
                // min = 0 (rand()%(max + 1 - min) + min)
                // no plus 1 as starting from 0 and size = SIZE_X/Y
                b = 1;
                
                while(b){ // while unoccupied position not found
                    b = 0;
                    
                    p_initial[k][i].coord.x = rand()%(SIZE_X);
                    p_initial[k][i].coord.y = rand()%(SIZE_Y);
                    
                    if(grid[p_initial[k][i].coord.x][p_initial[k][i].coord.y] == 1){ // already occupied
                        b = 1;
                    } else {
                        grid[p_initial[k][i].coord.x][p_initial[k][i].coord.y] = 1; // set grid cell occupied
                    }
                }
                
                // set agent heading values randomly (north, south, west, east)
                int directions[2] = {1, -1};
                int randInd = rand() % 2;
                
                if((double)rand()/RAND_MAX < 0.5){
                    p_initial[k][i].heading.x = directions[randInd];
                    p_initial[k][i].heading.y = 0;
                }
                else {
                    p_initial[k][i].heading.x = 0;
                    p_initial[k][i].heading.y = directions[randInd];
                }
            }
        }
        
        for(ind=0;ind<POP_SIZE;ind++){ // population level (iterate through individuals)
            
            // fitness evaluation - initialisation based on case
            if(FIT_EVAL == MIN){ // MIN - initialise to value higher than max
                fitness[ind] = (float)(SENSORS+1);
                tmp_fitness = (float)(SENSORS+1);
            }
            else { // MAX, AVG - initialise to zero
                fitness[ind] = 0.0;
                tmp_fitness = 0.0;
            }
            
            // reset prediction storage
            memset(pred, 0, sizeof(pred));

            // repetitions
            for(rep=0;rep<REPETITIONS;rep++){
                store = 0;
                
                tmp_fitness = doRun(gen, ind, p_initial[rep], MAX_TIME, 0, NUM_AGENTS);
                
                if(FIT_EVAL == MIN){ // min fitness of repetitions kept
                    if(tmp_fitness < fitness[ind]){
                        fitness[ind] = tmp_fitness;
                        
                        for(int s = 0; s<SENSORS; s++){ // average sensor predictions
                            pred[s] = pred_return[s];
                        }
                        store = 1;
                    }
                }
                else if(FIT_EVAL == MAX){ // max fitness of Repetitions kept
                    if(tmp_fitness > fitness[ind]){
                        fitness[ind] = tmp_fitness;
                        for(int s = 0; s<SENSORS; s++){
                            pred[s] = pred_return[s];
                        }

                        store = 1;
                    }
                }
                else if(FIT_EVAL == AVG){ // average fitness of Repetitions
                    fitness[ind] += (float)(tmp_fitness/REPETITIONS);
                    for(int s = 0; s<SENSORS; s++){
                        pred[s] += (float)pred_return[s]/(float)REPETITIONS;
                    }
                    
                    if(rep == REPETITIONS-1){ // store data of last repetition
                        store = 1;
                    }
                }
                
                // store best fitness + id of repetition
                if(store){
                    max_rep = rep;
                    
                    for(i=0; i<NUM_AGENTS; i++){ // store agent end positions
                        tmp_agent_maxfit_final[i].coord.x = p[i].coord.x;
                        tmp_agent_maxfit_final[i].coord.y = p[i].coord.y;
                        tmp_agent_maxfit_final[i].heading.x = p[i].heading.x;
                        tmp_agent_maxfit_final[i].heading.y = p[i].heading.y;
                        tmp_agent_maxfit_final[i].type = p[i].type;
                    }
                    
                    for(i=0; i<NUM_AGENTS; i++){  // store action values of best try of repetition
                        for(j=0; j<MAX_TIME; j++){
                            tmp_action[i][j] = current_action[i][j];
                        }
                    }
                }
            } // repetitions
            
            // average fitness of generation
            avg += fitness[ind];
            
            // store individual with maximum fitness
            if(fitness[ind]>max){
                max = fitness[ind];
                maxID = ind;
                
                // store agent predictions
                for(int s = 0; s<SENSORS; s++){
                    agentPrediction[s] = pred[s];
                }
                
                // store initial and final agent positions
                for(i=0; i<NUM_AGENTS; i++){
                    agent_maxfit[i].coord.x = tmp_agent_maxfit_final[i].coord.x;
                    agent_maxfit[i].coord.y = tmp_agent_maxfit_final[i].coord.y;
                    agent_maxfit[i].heading.x = tmp_agent_maxfit_final[i].heading.x;
                    agent_maxfit[i].heading.y = tmp_agent_maxfit_final[i].heading.y;
                    agent_maxfit[i].type = tmp_agent_maxfit_final[i].type;

                    agent_maxfit_beginning[i].coord.x = p_initial[max_rep][i].coord.x;
                    agent_maxfit_beginning[i].coord.y = p_initial[max_rep][i].coord.y;
                    agent_maxfit_beginning[i].heading.x = p_initial[max_rep][i].heading.x;
                    agent_maxfit_beginning[i].heading.y = p_initial[max_rep][i].heading.y;
                }
                
                // store action values of best run in generation
                for(j=0; j<NUM_AGENTS; j++){
                    for(i=0; i<MAX_TIME; i++){
                        action_values[j][i] = tmp_action[j][i];
                    }
                }
            }
        } // populations
        
        // print results: Generation, max fitness, max movement, id
        printf("#%d %f (%d)\n", gen, max, maxID);
        
        // write to files
        f = fopen(fit_file, "a");
        // write size of grid, generation, maximum fitness, average fitness,
        // maximum movement, average movement, sensor value predictions
        fprintf(f, "%d %d %d %e %e (%d) ",
                SIZE_X, SIZE_Y, gen, max, avg/(float)POP_SIZE, maxID);
        
        for(i=0; i<SENSORS; i++){
            fprintf(f, "%f ", agentPrediction[i]);
        }
    
        fprintf(f, "\n");
        fclose(f);
        
        f = fopen(agent_file, "a");
        fprintf(f, "Gen: %d\n", gen);
        fprintf(f, "Grid: %d, %d\n", SIZE_X, SIZE_Y);
        fprintf(f, "Fitness: %f \n", max);
        
        for(i=0; i<NUM_AGENTS; i++){
            fprintf(f, "%d, %d, %d, %d, %d, %d, %d, %d, %d\n", agent_maxfit[i].coord.x, agent_maxfit[i].coord.y,
                    agent_maxfit_beginning[i].coord.x, agent_maxfit_beginning[i].coord.y, agent_maxfit[i].heading.x,
                    agent_maxfit[i].heading.y, agent_maxfit_beginning[i].heading.x,
                    agent_maxfit_beginning[i].heading.y, agent_maxfit[i].type);
        }
        
        fprintf(f, "\n");
        fclose(f);
        
        f = fopen(actVal_file, "a");
        fprintf(f, "Gen: %d\n", gen);
        fprintf(f, "Grid: %d, %d\n", SIZE_X, SIZE_Y);
        fprintf(f, "Fitness: %f \n", max);
        
        for(i=0; i<NUM_AGENTS; i++){
            fprintf(f, "Agent: %d\n", i);
            fprintf(f, "[");
            for(j=0; j<MAX_TIME; j++){
                fprintf(f, "%d, ", action_values[i][j]);
            }
            fprintf(f, "]\n");
        }
        
        fprintf(f, "\n");
        fclose(f);
        
        // store/write to file genomes
        f = fopen(actGen_file, "a");
        for(i=0; i<LAYERS; i++){
            for(j=0; j<CONNECTIONS; j++){
                fprintf(f, "%f ", weight_actionNet[maxID][i][j]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
        fclose(f);
        
        f = fopen(predGen_file, "a");
        for(i=0; i<LAYERS; i++){
            for(j=0; j<CONNECTIONS; j++){
                fprintf(f, "%f ", weight_predictionNet[maxID][i][j]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
        fclose(f);
        
        // selection and mutation for evolution
        selectAndMutate(maxID, fitness);
    } // generations
    
    // re-run last / best controller of evolutionary run
    doRun(gen, maxID, agent_maxfit_beginning, MAX_TIME, 1, NUM_AGENTS);
    
    COUNT++;
}

/* main function
 * evolution or rerun of genome
 */
int main(int argc, char** argv){
    time_t *init_rand;
    int i = 0;
    int j = 0;
    int type1 = NOTYPE;
    
    // position of agents, current action value
    p = (struct agent*)malloc(NUM_AGENTS*sizeof(struct agent));
    p_next = (struct agent*)malloc(NUM_AGENTS*sizeof(struct agent));
    
    // initialise random number generator
    init_rand = malloc(sizeof(time_t));
    srand((unsigned int)time(init_rand));
    free(init_rand);
    
    // Replay
    if(!strcmp(argv[1], "REPLAY")){
        printf("REPLAY.\n");
        EVOL = 0;
        if(argc != 16){
            printf("Please specify 15 input values: [REPLAY/EVOL] [path to agent_trajectory file] [path to prediction_weights file] [path to action_weights file] [Fitness Function] [Manipulation: NONE/PRE/MAN] [Agent Type][REMOVE][DESTROY][New # agents][# of agents to be copied][x_min (destroy)][x_max][y_min][y_max] \n");
            exit(0);
        }
    }
    else if(!strcmp(argv[1], "EVOL")){
        printf("EVOLUTION.\n");
        EVOL = 1;
        if(argc != 7){
            // initialisation of variables via arguments
            printf("Please specify 6 input values: [REPLAY/EVOL] [Grid size x direction] [Grid size y direction] [Fitness Function] [Manipulation] [Agent Type (Manipulation)]\n");
            exit(0);
        }
    }
    else{ // no valid option
        printf("Please specify either REPLAY or EVOL as the first option.\n");
        exit(0);
    }

    if(EVOL){
        // Grid size
        SIZE_X = atof(argv[2]);
        SIZE_Y = atof(argv[3]);
    
        // set fitness function
        if(!strcmp(argv[4], "PRED")){
            printf("Fitness function: prediction\n");
            FIT_FUN = PRED;
        }
        else {
            printf("No valid fitness function specified.\n");
            exit(0);
        }

        // set manipulation value
        if(!strcmp(argv[5], "NONE")){
            printf("Manipulation: None\n");
            MANIPULATION = NONE;
        }
        else if(!strcmp(argv[5], "MAN")){
            printf("Manipulation: Manipulation\n");
            MANIPULATION = MAN;
        }
        else if(!strcmp(argv[5], "PRE")){
            printf("Manipulation: Predefined\n");
            MANIPULATION = PRE;
        }
        else{
            printf("No valid manipulation option specified.\n");
            exit(0);
        }
    
        if(MANIPULATION != NONE){
        
            // type1 - 1st evolutionary run
            if(!strcmp(argv[6], "LINE")){
                type1 = LINE;
            }
            else if(!strcmp(argv[6], "PAIR")){
                type1 = PAIR;
            }
            else if(!strcmp(argv[6], "DIAMOND")){
                type1 = DIAMOND;
            }
            else if(!strcmp(argv[6], "SQUARE")){
                type1 = SQUARE;
            }
            else{
                printf("Unknown Agent Type.\n");
                exit(0);
            }
        }
        
        // set agents to chosen type - set to NOTYPE if no manipulation
        for(i=0; i<NUM_AGENTS; i++){
            p[i].type = type1;
            p_next[i].type = type1;
        }
    
        printf("Sensor type: %d\n", SENSOR_MODEL);
        printf("Grid size = [%d, %d] \n", SIZE_X, SIZE_Y);
    
        // do evolutionary run
        evolution(NUM_AGENTS); // agents using best genome, number of genome set / evolutionary run
    }
    
    else{
        // do replay

        FILE *file;
        char line[2500];
        int i ;
        char *pt;
        int NUM_AGENTS_OLD = 100;
        struct agent initial[NUM_AGENTS_OLD];
        
        // set fitness function
        if(!strcmp(argv[5], "PRED")){
            printf("Fitness function: prediction\n");
            FIT_FUN = PRED;
        }
        else {
            printf("No valid fitness function specified.\n");
            exit(0);
        }
        
        // set manipulation value
        if(!strcmp(argv[6], "NONE")){
            printf("Manipulation: None\n");
            MANIPULATION = NONE;
        }
        else if(!strcmp(argv[6], "MAN")){
            printf("Manipulation: Manipulation\n");
            MANIPULATION = MAN;
        }
        else if(!strcmp(argv[6], "PRE")){
            printf("Manipulation: Predefined\n");
            MANIPULATION = PRE;
        }
        else{
            printf("No valid manipulation option specified.\n");
            exit(0);
        }
        
        // set agent type; if Manipulation == NONE default value of NOTYPE will be set
        if(MANIPULATION != NONE){
            if(!strcmp(argv[7], "LINE")){
                type1 = LINE;
            }
            else if(!strcmp(argv[7], "PAIR")){
                type1 = PAIR;
            }
            else if(!strcmp(argv[7], "DIAMOND")){
                type1 = DIAMOND;
            }
            else if(!strcmp(argv[7], "SQUARE")){
                type1 = SQUARE;
            }
            else{
                printf("Unknown Agent Type.\n");
                exit(0);
            }
        }
        
        /* INITIAL AGENT POSITIONS AND HEADINGS */
        printf("LOADING INITIAL AGENT POSITIONS...\n");
        file = fopen (argv[2], "r");
        
        if (file != NULL) {
            // skip first line
            fgets(line, 34,file);
            //get grid size
            fgets(line, 34,file);
            pt = strtok (line,": ");
            pt = strtok (NULL, ", ");
            SIZE_X = atoi(pt);
            pt = strtok (NULL, ", ");
            SIZE_Y = atoi(pt); 

            //compare agent number number with set one
            fgets(line, 34,file);
            pt = strtok (line,": ");
            pt = strtok (NULL," ");
            if(!(atoi(pt) == NUM_AGENTS_OLD)){
                printf("Please adjust NUM_AGENTS_OLD.\n");
                exit(0);
            }

            //get agent positions and headings
            for(i = 0; i < NUM_AGENTS_OLD;i++) {
                //fseek(datei, i ,SEEK_SET);
                fgets(line, 34,file);  // read next line
                char *pt;
                pt = strtok (line,": ");
                pt = strtok (NULL,", ");
                
                initial[i].coord.x = atoi(pt);
                pt = strtok (NULL,", ");
                initial[i].coord.y = atoi(pt);
                pt = strtok (NULL,", ");
                initial[i].heading.x = atoi(pt);
                pt = strtok (NULL,", ");
                initial[i].heading.y = atoi(pt);
            }
            printf("COMPLETED.\n");
            fclose (file);
        }
        else{
            printf("PATH ERROR.\n");
            exit(0);
        }
        
        /* PREDICTION NETWORK WEIGHTS */
        if(MANIPULATION != PRE){
            printf("LOADING PREDICTION GENOME...\n");

            file = fopen (argv[3], "r");
            if (file != NULL) {
                i = 0;
                while (fgets(line, sizeof(line), file)) {
                    i++;
                    if(i == 397){
                        break;
                    }
                }
            
                // first line weights
                pt = strtok (line,": ");
                for(i=0; i<CONNECTIONS; i++){
                    weight_predictionNet[0][0][i] = atof(pt);
                    pt = strtok (NULL,": ");
                }
            
                // second line weights
                fgets(line, sizeof(line), file);
                pt = strtok (line,": ");
                for(i=0; i<CONNECTIONS; i++){
                    weight_predictionNet[0][1][i] = atof(pt);
                    pt = strtok (NULL,": ");
                }
                  
                // third line weights
                fgets(line, sizeof(line), file);
                pt = strtok (line,": ");
                for(i=0; i<CONNECTIONS; i++){
                    weight_predictionNet[0][2][i] = atof(pt);
                    pt = strtok (NULL,": ");
                }
            
                printf("COMPLETED.\n");
                fclose (file);
            }
            else{
                printf("PATH ERROR.\n");
                exit(0);
            }
        }
        
        /* ACTION NETWORK WEIGHTS */
        printf("LOADING ACTION GENOME...\n");

        file = fopen (argv[4], "r");
        if (file != NULL) {
            i = 0;
            while (fgets(line, sizeof(line), file)) {
                i++;
                if(i == 397){
                    break;
                }
            }
            
            // first line weights
            pt = strtok (line,": ");
            for(i=0; i<CONNECTIONS; i++){
                weight_actionNet[0][0][i] = atof(pt);
                pt = strtok (NULL,": ");
            }
            
            // second line weights
            fgets(line, sizeof(line), file);
            pt = strtok (line,": ");
            for(i=0; i<CONNECTIONS; i++){
                weight_actionNet[0][1][i] = atof(pt);
                pt = strtok (NULL,": ");
            }
            
            // third line weights
            fgets(line, sizeof(line), file);
            pt = strtok (line,": ");
            for(i=0; i<CONNECTIONS; i++){
                weight_actionNet[0][2][i] = atof(pt);
                pt = strtok (NULL,": ");
            }
            
            printf("COMPLETED.\n");
            fclose (file);
        }
        else{
            printf("PATH ERROR.\n");
            exit(0);
        }
        
        // set agents to chosen type - set to NOTYPE if no manipulation - BEGINNING: only no Manipulation cases replay
        for(i=0; i<NUM_AGENTS; i++){
            p[i].type = type1;
            p_next[i].type = type1;
        }

        // write fitness values to file
        FILE *f;
        f = fopen("replay_fitness", "a");

        //re-run genome with initial position before - output position last time step from initial run 
        fprintf(f, "Fitness of re-run: %e\n", doRun(0, 0, initial, MAX_TIME, 1, NUM_AGENTS_OLD));
        
        for(i=0; i<SENSORS; i++){
            fprintf(f, "%f ", pred_return[i]);
        }
        fprintf(f, "\n");
        
        COUNT = -1; // change count number to indicate self-repair run
        
        int REMOVE = atoi(argv[8]);
        int DESTROY = atoi(argv[9]);
        int new_agent_no = atoi(argv[10]);
        int copy_agent_no = atoi(argv[11]);
        int grid[SIZE_X][SIZE_Y];
        int x_min = 0;
        int x_max = 0;
        int y_min = 0;
        int y_max = 0;
        
        memset(grid, 0, sizeof(grid));
        
        if(DESTROY){ // define area in which agents will be removed
            x_min = atoi(argv[12]);
            x_max = atoi(argv[13]);
            y_min = atoi(argv[14]);
            y_max = atoi(argv[15]);
            
            copy_agent_no = NUM_AGENTS_OLD; // only agents in defined area won't be copied
        }
        
        struct agent replay[new_agent_no];
        
        if(!DESTROY){
            
            // copy last position of agents
            // random starting position for copying agents
            int start = rand()%(NUM_AGENTS_OLD);
            for(i=0; i<copy_agent_no; i++){
                
                if (start+i >= NUM_AGENTS_OLD){
                    start = -i;
                }
                                
                replay[i].coord.x = p[start+i].coord.x;
                replay[i].coord.y = p[start+i].coord.y;
                replay[i].heading.x = p[start+i].heading.x;
                replay[i].heading.y = p[start+i].heading.y;
                
                grid[replay[i].coord.x][replay[i].coord.y] = 1;
            }
            
            // initialise agent positions to random discrete x & y values
            // min = 0 (rand()%(max + 1 - min) + min)
            // no plus 1 as starting from 0 and size = SIZE_X/Y
            for(i=copy_agent_no; i<new_agent_no; i++){
                int b = 1;
                
                while(b){
                    b = 0;
                    
                    replay[i].coord.x = rand()%(SIZE_X);
                    replay[i].coord.y = rand()%(SIZE_Y);
                    
                    if(grid[replay[i].coord.x][replay[i].coord.y] == 1){
                        b = 1;
                    } else {
                        grid[replay[i].coord.x][replay[i].coord.y] = 1;
                    }
                }
                
                // set agent heading values randomly (north, south, west, east possible)
                int directions[2] = {1, -1};
                int randInd = rand() % 2;
                
                if((double)rand()/RAND_MAX < 0.5){
                    replay[i].heading.x = directions[randInd];
                    replay[i].heading.y = 0;
                }
                else {
                   replay[i].heading.x = 0;
                   replay[i].heading.y = directions[randInd];
                }
            }
        }
        else{
            printf("DESTROY.\n");
            
            // Remove agents in a certain area
            
            // set all grid values as 1 where agents should be removed to not position new agents there when adding agents randomly
            for(i = x_min; i <= x_max; i++){
                for(j = y_min; j <= y_max; j++){
                    grid[i][j] = 1;
                }
            }
            
            j = 0;
            
            // copy agents
            for(i=0; i<copy_agent_no; i++){

                // copy only if agent not positioned within part to be removed
                if(!((p[i].coord.x <= x_max && p[i].coord.x >= x_min) && (p[i].coord.y <= y_max && p[i].coord.y >= y_min))){
                    replay[j].coord.x = p[i].coord.x;
                    replay[j].coord.y = p[i].coord.y;
                    replay[j].heading.x = p[i].heading.x;
                    replay[j].heading.y = p[i].heading.y;
                    
                    grid[replay[j].coord.x][replay[j].coord.y] = 1;
                    
                    j++;
                }
            }
            
            if(!REMOVE){ // if not remove - replace agents to new positions 
                for(j=j; j<new_agent_no; j++){
                    int b = 1;
                    
                    while(b){
                        b = 0;
                        
                        replay[j].coord.x = rand()%(SIZE_X);
                        replay[j].coord.y = rand()%(SIZE_Y);
                        
                        if(grid[replay[j].coord.x][replay[j].coord.y] == 1){
                            b = 1;
                        }
                        else {
                            grid[replay[j].coord.x][replay[j].coord.y] = 1;
                        }
                    }
                    
                    // set agent heading values randomly (north, south, west, east possible)
                    int directions[2] = {1, -1};
                    int randInd = rand() % 2;
                    
                    if((double)rand()/RAND_MAX < 0.5){
                        replay[j].heading.x = directions[randInd];
                        replay[j].heading.y = 0;
                    }
                    else {
                        replay[j].heading.x = 0;
                        replay[j].heading.y = directions[randInd];
                    }
                }
            }
            
            // adjust agent number 
            new_agent_no = j;
        }
        
        fprintf(f, "Number of agents: %d\n", new_agent_no);
        if(!DESTROY){
            fprintf(f, "Number of agents on same starting position: %d\n", copy_agent_no);
        }
        
        fprintf(f, "Fitness of self-repair: %e\n", doRun(0, 0, replay, MAX_TIME, 1, new_agent_no));
        
        for(i=0; i<SENSORS; i++){
            fprintf(f, "%f ", pred_return[i]);
        }
        fprintf(f, "\n");
        
        fclose(f);
    }
    
    free(p);
    free(p_next);
}

