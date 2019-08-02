//
//  STD14MANL.h
//  minimizeSurpriseSelfAssembly
//
//  Created by Tanja Kaiser on 14.03.18.
//  Copyright © 2018 Tanja Kaiser. All rights reserved.
//
#include "STD.h"

#ifndef STD14MANL_h
#define STD14MANL_h

// ANN parameter
#define LAYERS 3  // ANN layers

#define INPUTA 15    // input action network (14 sensors + 1 action value)
#define HIDDENA  8  // hidden action network
#define OUTPUTA  2  // output action network

#define INPUTP  15   // input prediction network (14 sensors + 1 action value)
#define HIDDENP  12 // hidden prediction network
#define OUTPUTP  10 // output prediction network (14 sensor predictions)

#define CONNECTIONS  192 // maximum connections

#define SENSORS 14

#define SENSOR_MODEL STDL 

#endif /* STD14MANL_h */

