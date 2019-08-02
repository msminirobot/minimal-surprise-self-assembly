//
//  STD8.h
//  minimizeSurpriseSelfAssembly
//  Moore Neighborhood 
//
//  Created by Tanja Kaiser on 08.08.18.
//  Copyright © 2018 Tanja Kaiser. All rights reserved.
//
#include "STD.h"

#ifndef STD8_h
#define STD8_h

// ANN parameter
#define LAYERS 3  // ANN layers

#define INPUTA 9  // input action network (8 sensors + 1 action value)
#define HIDDENA  6 // hidden action network
#define OUTPUTA  2 // output action network

#define INPUTP  9  // input prediction network (8 sensors + 1 action value)
#define HIDDENP  8 // hidden prediction network
#define OUTPUTP  8 // output prediction network (8 sensor predictions)

#define CONNECTIONS  80 // maximum connections (80 per layer: 1 per INPUTP for each HIDDENP (9x8) + recurrent --> 8 extra values)

#define SENSORS 8

#define SENSOR_MODEL STDSL

#endif /* STD8_h */
