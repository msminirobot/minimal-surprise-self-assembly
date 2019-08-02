//
//  STD6.h
//  minimizeSurpriseSelfAssembly
//
//  Created by Tanja Kaiser on 14.03.18.
//  Copyright © 2018 Tanja Kaiser. All rights reserved.
//

#ifndef STD6_h
#define STD6_h

// ANN parameter
#include "STD.h"

#define LAYERS 3  // ANN layers

#define INPUTA   7   // input action network (6 sensors + 1 action value)
#define HIDDENA  4  // hidden action network
#define OUTPUTA  2 // output action network

#define INPUTP   7   // input prediction network (6 sensors + 1 action value)
#define HIDDENP  6  // hidden prediction network
#define OUTPUTP  6 // output prediction network (6 sensor predictions)

#define CONNECTIONS 48 // maximum connections (48 per layer: 1 per INPUTP for each HIDDENP (15x14) + recurrent --> 14 extra values)

#define SENSORS 6

#define SENSOR_MODEL STDS

#endif /* STD6_h */

