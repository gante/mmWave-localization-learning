//---------------------------------------------------------------------------//
//	     João Gante's PhD work                         IST-Portugal          //
//                                 Oct 2017                                  //
//                                                                           //
//        General includes(hpp) - All the main includes will be here         //
//       (if used by more than one pair of cpp/hpp, will also be here)       //
//---------------------------------------------------------------------------//

#ifndef INC_GENERAL_H
#define INC_GENERAL_H


#include <iostream>
#include <cmath>

//#include <cstdio>
//#include <fstream>

#include <cstring>
//#include <ctime>
//#include <chrono>
//#include <stdlib.h>

using namespace std;


//Ray-tracing parameters
#define RESOLUTION 1.0
#define GRID_SIZE_X 400.0
#define GRID_SIZE_Y 400.0
#define BEAMFORMINGS 32

//Preprocessing parameters
#define SAMPLE_FREQ 20000000.0   // default: 20 MHz = 1 sample per 0.05 us
#define MAX_DELAY 0.000006      // = 6 us (slowest signal = 6.2 us, only 255 com 6+ us)
#define MIN_POWER -150.0        // minimum power = -150 dBm

//CURRENT MAP = NYU
#define STARTING_X -183.0
#define STARTING_Y -176.0




#endif /* INC_GENERAL_H */

