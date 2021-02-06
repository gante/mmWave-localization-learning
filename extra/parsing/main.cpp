//---------------------------------------------------------------------------//
//	     João Gante's PhD work                         IST-Portugal          //
//                                 Oct 2017                                  //
//                                                                           //
//           Data Preprocessing - transforms the discrete impulses           //
//                                into fixed time intervals values           //
//---------------------------------------------------------------------------//


// ~~~~~~~~~~~~~~~   IMPORTANT   ~~~~~~~~~~~~~~~~~~

// If using too much RAM (> 1 GB), change the processing
// (i.e. remove the BEAMFORMING dimension from the table
// and store after each file)

// ~~~~~~~~~~~~~~~   IMPORTANT   ~~~~~~~~~~~~~~~~~~





#include "general_includes.hpp"


void create_position_table(const int total_elements);
void process_paths(const int & paths, const int & time_slots, const float path_phase[1000],
                   const float path_delay[1000], const float path_power[1000], float * slot_values);
void copy_to_main_data(const float * slot_values, float * data_table, const int & i,
                       const int & bf_index, const int & time_slots);
int flag_invalid(float * data_table, const int & users, const int & time_slots);
void fill_final_table(float * final_table, const int & valid_users, const int & elements_per_user);


float dbm_to_mw(const float & x);
float mw_to_dbm(const float & x);




////////////////////////////////////////////////////////////////////////////////////
// Function: Main
// Details: The main function over here
// Inputs: --
// Outputs: --
////////////////////////////////////////////////////////////////////////////////////
int main(){

    //Sequence:
    // 1 - Creates the tables [#user vs position   & main table]
    // 2 - Setting the outer loop, each iteration corresponds to a CIR file
    // 3 - Runs the inner loop, each iteration corresponds to a user
    //      a) Reads the user header (#user, total paths)
    //      b) Reads all the paths
    //      c) Distributes the paths among the time intervals
    // 4 - Checks for valid/invalid positions (invalid has a +1 at the first value)
    // 5 - Stores the resulting table (and cleans up)
    // 6 - Converts the two tables into DNN-ready data




    //--------------------------------------------------------------------------//
    // 1 - Creates the positions table [#user vs position]                      //
    //--------------------------------------------------------------------------//


    //Knowing the problem size (area + resolution), it's easy to create the tables

    int X_elements = (GRID_SIZE_X/RESOLUTION) + 1;
    int Y_elements = (GRID_SIZE_Y/RESOLUTION) + 1;
    int total_elements = X_elements * Y_elements;


    //Creates, initializes and stores the position table
    create_position_table(total_elements);


    //Table format: dim1 = N_elements = X_elements * Y_elements
    //              dim2 = beamformings = BEAMFORMINGS
    //              dim3 = time slots = MAX_DELAY / (1/SAMPLE_FREQ)

    cout << "Initializing the data table...";

    int time_slots = int(MAX_DELAY / (1/SAMPLE_FREQ) );

    float * data_table;
    data_table = new float[total_elements * BEAMFORMINGS * time_slots];


    cout << " Done!\n\n" << endl;



    //--------------------------------------------------------------------------//
    // 2 - Setting the outer loop, each iteration corresponds to a CIR file     //
    //--------------------------------------------------------------------------//


    int bf_index, i, j, user, paths;
    char filename[100], buffer[1000];
    float path_power[1000], path_phase[1000], path_delay[1000], f_buffer;
    float * slot_values;
    FILE *InputFile;

    slot_values = new float[time_slots];


    for(bf_index = 1; bf_index <= BEAMFORMINGS; bf_index++){

        //Loads the file, with the name template "\\data_raw\\cir_" + i + ".txt"
        sprintf(filename, "data_raw/CIR_32/cir_%d.txt", bf_index);
        InputFile = fopen(filename, "r");

        cout << "Processing file #" << bf_index << "... ";

        //If there was any problem with the file loading, sets the error up and breaks
        if(!InputFile) {
            cout << "ERROR loading a simulation file (" << bf_index << ")" << endl;
            exit(2);
        }

        //Ignores the first three lines (unneeded information)
        for(i = 0; i < 3; i++){
            fgets(buffer, sizeof(buffer), InputFile);
        }



    //--------------------------------------------------------------------------//
    // 3 - Runs the inner loop, each iteration corresponds to a user            //
    //--------------------------------------------------------------------------//



        // For each user, the processing is split among two sections
        // i)   user header (1 line, indicates how many paths)
        // ii)  user paths  (N lines, with the key information)


        for(i = 1; i <= total_elements; i++){

            //      a) Reads the user header (#user, total paths)

            //Reads the user index
            if(fscanf(InputFile,"%d",&user) == 0){
                cout << "\nERROR reading a value";
                exit(3);
            }
            //Reads the # of paths for that user
            if(fscanf(InputFile,"%d",&paths) == 0){
                cout << "\nERROR reading a value";
                exit(3);
            }


            if(user != i){
                cout << "\nERROR double checking the user index";
                exit(4);
            }

            if(paths > 999){
                cout << "\nERROR too many paths! (increase the buffers)";
                exit(5);
            }



            //      b) Reads all the paths
            for(j = 0; j < paths; j++){

                //Reads the path index (not used)
                if(fscanf(InputFile,"%f",&f_buffer) == 0){
                    cout << "\nERROR reading a value";
                    exit(3);
                }

                //Reads the path phase
                if(fscanf(InputFile,"%f",&f_buffer) == 0){
                    cout << "\nERROR reading a value";
                    exit(3);
                }
                path_phase[j] = f_buffer;

                //Reads the path delay
                if(fscanf(InputFile,"%f",&f_buffer) == 0){
                    cout << "\nERROR reading a value";
                    exit(3);
                }
                path_delay[j] = f_buffer;

                //Reads the path power in dBm
                if(fscanf(InputFile,"%f",&f_buffer) == 0){
                    cout << "\nERROR reading a value";
                    exit(3);
                }
                path_power[j] = f_buffer;

            } // end the *for* cycle, which goes through all the paths


            //      c) Distributes the paths among the time intervals
            process_paths(paths, time_slots, path_phase, path_delay, path_power, slot_values);
            copy_to_main_data(slot_values, data_table, i, bf_index, time_slots);


            if(i%(int)(total_elements/10) == 0){
                cout << "*";
            }

        } // end the *for* cycle, which goes through all the users

        fclose(InputFile);

        cout << " Done!" << endl;

    } // end the *for* cycle, which goes through all the files



    //--------------------------------------------------------------------------//
    // 4 - Checks for valid positions (invalid has a +1 at the first value)     //
    //--------------------------------------------------------------------------//

    cout << "Flagging invalid positions...";
    int invalid_postions;
    invalid_postions = flag_invalid(data_table, total_elements, time_slots);
    cout << " Done!" << endl;


    //--------------------------------------------------------------------------//
    // 5 - Stores the resulting table (and cleans up)                           //
    //--------------------------------------------------------------------------//

    cout << "Storing the results...";

    FILE *Data;

    //Opens the data file
    sprintf(filename, "./data_processed/data_table");
    Data = fopen(filename, "wb");

    if(!Data) {
        cout << "\nERROR opening the data file";
        exit(2);
    }

    //Writes the data
    fwrite(data_table, sizeof(float), (total_elements * BEAMFORMINGS * time_slots), Data);

    fclose(Data);


    delete [] data_table;
    delete [] slot_values;

    cout << " Done!" << endl;



    //--------------------------------------------------------------------------//
    // 6 - Converts the two tables into DNN-ready data                          //
    //--------------------------------------------------------------------------//

    cout << "Converting into the final table...";

    //Final table : pos_0, data_pos_0, pos_1, data_pos_1,...
    //              (discards the invalid positions)
    //              Current data format: 32 bit floating point

    int valid_users = (total_elements-invalid_postions);
    int elements_per_pos = ((BEAMFORMINGS * time_slots) + 2);
    int final_data_elements = valid_users * elements_per_pos;

    float * final_table;
    final_table = new float[final_data_elements];

    //fills the table
    fill_final_table(final_table,valid_users,elements_per_pos);

    //stores the table
    sprintf(filename, "./data_processed/final_table");
    Data = fopen(filename, "wb");

    if(!Data) {
        cout << "\nERROR opening the final data file";
        exit(2);
    }

    //Writes the data
    fwrite(final_table, sizeof(float), final_data_elements, Data);

    fclose(Data);

    delete [] final_table;

    cout << " Done!" << endl;


    cout << "\nSuccess!" << endl;

    return(0);
}





////////////////////////////////////////////////////////////////////////////////////
// Function: Create Position Table
// Details: Creates, initializes and stores the position table
// Inputs: Total number of elements (the rest are defines)
// Outputs: --void fill_final_table(float * final_table, const int & valid_users, const int & elements_per_user)
////////////////////////////////////////////////////////////////////////////////////
void create_position_table(const int total_elements){

    //Position Table:   correspondence between position index and actual position [#user = implicit, X, Y]

    cout << "\nInitializing the position table...";

    int i, index;
    float x_aux = 0.0, y_aux = 0.0;

    float * position_table;
    position_table = new float[total_elements * 2];

    //initializes the position table data
    // (the users are numbered in a "index = x + y*Y_elements" fashion)
    for(i = 0; i < total_elements; i++){
        index = i*2;

        position_table[index] = STARTING_X + x_aux;     // X
        position_table[index+1] = STARTING_Y + y_aux;   // Y

        x_aux = x_aux + RESOLUTION;
        if(x_aux > GRID_SIZE_X){
            x_aux = 0.0;
            y_aux = y_aux + RESOLUTION;
        }
    }


    //Opens the position file
    char filename[100];
    FILE *OutputFile;

    sprintf(filename, "./data_processed/position_table");
    OutputFile = fopen(filename, "wb");

    if(!OutputFile) {
        cout << "\nERROR opening the position file!";
        exit(1);
    }

    //Writes the data
    fwrite(position_table, sizeof(float), (total_elements*2), OutputFile);

    //clean up
    delete [] position_table;

    cout << " Done!" << endl;
}



////////////////////////////////////////////////////////////////////////////////////
// Function: Process paths
// Details: merges the paths phase, delay and power info into the correct time slot
// Inputs: # paths, paths' phase, delay and power information
// Outputs: the slot values (which then must be copied into the main matrix)
////////////////////////////////////////////////////////////////////////////////////
void process_paths(const int & paths, const int & time_slots, const float path_phase[1000],
                   const float path_delay[1000], const float path_power[1000], float * slot_values){

    int i, current_slot = 0, exist_path;

    float time_per_slot = 1/SAMPLE_FREQ;
    float current_time = 0.0;
    float next_time = current_time + time_per_slot;
    float tmp_phase, i_power, q_power, power_mw;


    // For each time slot, passes through all the paths
    while(current_slot < time_slots){

        for(i = 0; i < paths; i++){

            //checks if the path belongs to the desired time slot
            if(path_delay[i] >= current_time && path_delay[i] < next_time && path_power[i] >= MIN_POWER){

                //if it does, checks if there is another path for this slot
                //      no previous path = stores directly in dbm
                //      previous path = IQ processing
                if(exist_path == 0){
                    exist_path = 1;
                    slot_values[current_slot] = path_power[i];
                    tmp_phase = path_phase[i];
                }
                else{
                    //when processing the 2nd path, also converts the first into IQ
                    if(exist_path == 1){
                        exist_path = 2;
                        power_mw = dbm_to_mw(slot_values[current_slot]);
                        i_power = power_mw * cos(tmp_phase);
                        q_power = power_mw * sin(tmp_phase);
                    }

                    power_mw = dbm_to_mw(path_power[i]);
                    i_power += power_mw * cos(path_phase[i]);
                    q_power += power_mw * sin(path_phase[i]);

                }

            }

        }


        //if multiple paths were processed, converts the IQ into dBm
        if(exist_path == 2){
            //power_dbm = dbm( power_mw ), power_mw = sqrt(i^2 + q^2)
            power_mw = sqrt(pow(i_power,2) + pow(q_power,2));
            slot_values[current_slot] = mw_to_dbm(power_mw);

        }
        //if no path was processed, store 0
        else if(exist_path == 0){
            slot_values[current_slot] = 0.0;
        }


        //prepares the data for the next time_slot
        current_time = next_time;
        next_time = current_time + time_per_slot;
        current_slot++;
        exist_path = 0;
    }


    if( (current_slot - 1) > time_slots){
        cout << "ERROR! too many time slots stored";
        exit(6);
    }


}


////////////////////////////////////////////////////////////////////////////////////
// Function: Copy to main data
// Details: copies the processed path data into the main data
// Inputs: processed path data, BF index, user index
// Outputs: main data filled up
////////////////////////////////////////////////////////////////////////////////////
void copy_to_main_data(const float * slot_values, float * data_table, const int & user,
                       const int & bf_index, const int & time_slots){


    //data_table = new float[total_elements * BEAMFORMINGS * time_slots];
    // ATTENTION: both "user" and "bf_index" start at 1, so we should subtract it :D
    int i;
    int starting_index = ((user-1) *(BEAMFORMINGS * time_slots)) + ((bf_index-1) * (time_slots));

    for(i = 0; i < time_slots; i++){
        data_table[starting_index + i] = slot_values[i];
    }


}



////////////////////////////////////////////////////////////////////////////////////
// Function: Flag Invalid
// Details: Flags invalid positions with a "1" (= impossible value) at the first time slot
// Inputs: data size and the data
// Outputs: updated data, number of INVALID positions
////////////////////////////////////////////////////////////////////////////////////
int flag_invalid(float * data_table, const int & users, const int & time_slots){

    int i, j, index, slots_to_check = BEAMFORMINGS * time_slots, invalids = 0;
    bool valid, dbg = false;

    int dbg_40 = 0, dbg_50 = 0, dbg_60 = 0, dbg_index;

    //for each user, scans the BF * TS for non-zero values
    for(i = 0; i < users; i++){

        valid = false;
        index = i * (slots_to_check);


        //for user i, scans the j slots
        for(j = 0; j < slots_to_check; j++){

            //as soon as it finds a filled slot, breaks
            if(data_table[index + j] != 0.0){
                valid = true;

                if(dbg){
                    dbg_index = j%70;
                    if(dbg_index > 40) dbg_40++;
                    if(dbg_index > 50) dbg_50++;
                    if(dbg_index > 60) dbg_60++;
                }

                else{
                    break;
                }
            }

        }



        //if no valid was found, flags it
        if(valid == false){
            data_table[index] = 1;
            invalids++;
        }


        //dbg: prints user #14648 = index 14647
        if(dbg && i == 14647){

            cout << "\nDBG: printing user # 14648" << endl;

            for(j = 0; j < slots_to_check; j++){
                cout << "j = " << j << "; dBm = " << data_table[index + j] << endl;
            }

        }


    }

    cout << "(" << invalids << " invalid positions)";

    if(dbg){
        cout << "(" << dbg_40 << " = ts_40)";
        cout << "(" << dbg_50 << " = ts_50)";
        cout << "(" << dbg_60 << " = ts_60)";
    }

    return(invalids);
}



////////////////////////////////////////////////////////////////////////////////////
// Function: Fill Final Table
// Details: Fills the Final Table, based on previous data
// Inputs: empty table, number of valid users
// Outputs: filled table
////////////////////////////////////////////////////////////////////////////////////
void fill_final_table(float * final_table, const int & valid_users, const int & elements_per_pos){


    //          1 - Opens the files

    FILE *Data;
    FILE *Positions;
    char filename[100];

    //Opens the data file
    sprintf(filename, "./data_processed/data_table");
    Data = fopen(filename, "rb");

    if(!Data) {
        cout << "\nERROR opening the data file";
        exit(2);
    }


    //Opens the positions file
    sprintf(filename, "./data_processed/position_table");
    Positions = fopen(filename, "rb");

    if(!Positions) {
        cout << "\nERROR opening the positions file";
        exit(2);
    }



    //          2 - Main loop, where it reads 1 entry and decides
    //              whether it must save it or not

    int read_data_elements = elements_per_pos - 2, valid_count = 0, pos = 0;
    float * read_data, * read_position;
    read_data = new float[read_data_elements];
    read_position = new float[2];
    bool dbg = false;

    //while there are stuff to read, keeps going
    while(fread (read_data, sizeof(float), read_data_elements, Data) == (unsigned int)read_data_elements){


        //reads the correspondent position data
        fread(read_position, sizeof(float), 2, Positions);

        if(dbg && (pos == 0 || pos == 401 || pos == 801 || pos == 802) ){
            cout << "\nDBG: pos " << pos << " has the following location: X=" << read_position[0] << " Y=" << read_position[1];
        }


        //checks for invalid position
        if(read_data[0] != 1){

            //if it valid, copies the contents
            memcpy(final_table+(valid_count*elements_per_pos), read_position, 2 * sizeof(float));
            memcpy(final_table+(valid_count*elements_per_pos)+2, read_data, read_data_elements * sizeof(float));


            valid_count++;

        }


        pos++;

    }


    if(valid_count != valid_users){
        cout << "\nERROR - the number of valid users don't match! ";
        cout << valid_count;
        cout << valid_users;
    }

    delete [] read_data;
    delete [] read_position;
}



////////////////////////////////////////////////////////////////////////////////////
//quick tool: dbm to mw
////////////////////////////////////////////////////////////////////////////////////
float dbm_to_mw(const float & x){

    return(   pow(10, (x/10))  );

}

////////////////////////////////////////////////////////////////////////////////////
//quick tool: mw to dbm
////////////////////////////////////////////////////////////////////////////////////
float mw_to_dbm(const float & x){

    return(  10*log10(x)  );

}
