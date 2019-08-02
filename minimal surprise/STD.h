//
//  STD.h
//  minimizeSurpriseSelfAssembly
//
//  Created by Tanja Kaiser on 14.03.18.
//  Copyright © 2018 Tanja Kaiser. All rights reserved.
//

#ifndef STD_h
#define STD_h

// experimental setup
#define MUTATION 0.1 // 0.1 - mutation rate
#define POP_SIZE 50 // population

#define MAX_TIME 500  // time per run
#define MAX_GENS 100  // maximum generations

#define REPETITIONS 10 // repetitions of each individual
#define NUM_AGENTS 100

// movement
#define STRAIGHT 0
#define TURN 1
#define UP 1
#define DOWN -1

// sensors
#define S0 0 // forward
#define S1 1 // forward right
#define S2 2 // forward left
#define S3 3 // 2 cells forward
#define S4 4 // 2 cells forward right
#define S5 5 // 2 cells forward left
#define S6 6 // right of agent
#define S7 7 // left of agent
#define S8 8 // backward
#define S9 9 // backward right
#define S10 10 // backward left
#define S11 11 // 2 cells backward
#define S12 12 // 2 cells backward right
#define S13 13 // 2 cells backward left

#define PI 3.14159265

// fitness evaulation
#define MIN 0
#define MAX 1
#define AVG 2
#define FIT_EVAL MIN

// define fitness function
#define PRED 0   // prediction

// define sensor model
#define STDS 0 // standard: 6 sensors - in heading direction forward / right / left (1 & 2 blocks ahead)
#define STDL 2 // standard large: 14 sensors - 6 in heading direction (like STD), 2 next to agent, 6 behind agent
#define STDSL 3 // Moore Neighborhood 

// define manipulation models
#define NONE 0 // none
#define PRE 1 // prediction
#define MAN 2 // manipulation

// agent types
#define NOTYPE -1
#define LINE 1
#define DIAMOND 4
#define SQUARE 5
#define PAIR 6

#endif /* STD_h */
