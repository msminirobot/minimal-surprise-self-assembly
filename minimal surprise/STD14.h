//
//  STD14.h
//  minimizeSurpriseSelfAssembly
//
//  Created by Tanja Kaiser on 14.03.18.
//  Copyright © 2018 Tanja Kaiser. All rights reserved.
//
#include "STD.h"

#ifndef STD14_h
#define STD14_h

// ANN parameter
#define LAYERS 3  // ANN layers

#define INPUTA 15  // input action network (14 sensors + 1 action value)
#define HIDDENA  8 // hidden action network
#define OUTPUTA  2 // output action network

#define INPUTP  15  // input prediction network (14 sensors + 1 action value)
#define HIDDENP  14 // hidden prediction network
#define OUTPUTP  14 // output prediction network (14 sensor predictions)

#define CONNECTIONS  224 // maximum connections (224 per layer: 1 per INPUTP for each HIDDENP (15x14) + recurrent --> 14 extra values)

#define SENSORS 14

#define SENSOR_MODEL STDL 

#endif /* STD14_h */
