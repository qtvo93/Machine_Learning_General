/*
* Name: Quoc Thinh Vo
* Date: 09/30/2020

*Project Branch_Predictor
*Supervised Training & Perceptron Algorithm
*Part 1: TWO_BIT_LOCAL Predictor

+-------------------+------------------+---------+----------+---------------+ 
|LocalPredictorSize | LocalCounterBits | Correct |Incorrect | % Correctness | 
+-------------------+------------------+---------+----------+---------------+ 
| 2048 		    | 1                | 8217    | 1783     | 82.169998000  | 
| 2048              | 2                | 8467    | 1533     | 84.669998000  | 
| 4096              | 2                | 8468    | 1532     | 84.680000000  | 
| 8192              | 2 	       | 8476    | 1524     | 84.759995000  | 
| 16384             | 2 	       | 8474    | 1526     | 84.739998000  | 
| 32768             | 2 	       | 8473    | 1527     | 84.729996000  | 
| 65536             | 2 	       | 8473    | 1527     | 84.729996000  | 
+-------------------+------------------+---------+----------+---------------+

The best combination that give us the best performance is:
LocalPredictorSize : 8192 and LocalCounterBits: 2 with 84.759995% correctness.

*Part 2: TOURNAMENT Predictor:

+----------------------+--------------------+---------------------+----------+----------+---------------+ 
|LocalHistoryTableSize |GlobalPredictorSize | ChoicePredictorSize | Correct  |Incorrect | % Correctness | 
+----------------------+--------------------+---------------------+----------+----------+---------------+ 
| 2048 		       | 8192               |	8192		  | 8060     | 1940     | 80.599998000  | 
| 4096                 | 8192               |	8192		  | 8073     | 1928     | 80.720001000  | 
| 4096                 | 16384              |	16384		  | 8064     | 1936     | 80.639999000  | 
| 8192                 | 16384              |	16384		  | 8064     | 1936     | 80.639999000  | 
| 16384                | 32768 	            |	32768		  | 8046     | 1954     | 80.459999000  | 
+----------------------+--------------------+---------------------+----------+----------+---------------+

The best combination that give us the best performance is:
LocalHistoryTableSize : 4096 and GlobalPredictorSize = ChoicePredictorSize : 8192 with 80.720001% correctness.

*Part 3: GSHARE Predictor:

+----------------------+--------------------+---------------------+----------+----------+---------------+ 
|LocalHistoryTableSize |GlobalPredictorSize | ChoicePredictorSize | Correct  |Incorrect | % Correctness | 
+----------------------+--------------------+---------------------+----------+----------+---------------+ 
| 2048 		       | 8192               |	8192		  | 7485     | 2215     | 74.849998000  | 
| 4096                 | 8192               |	8192		  | 7485     | 2215     | 74.849998000  | 
| 4096                 | 16384              |	16384		  | 7422     | 2578     | 74.220001000  | 
| 8192                 | 16384              |	16384		  | 7422     | 2578     | 74.220001000  | 
| 16384                | 32768 	            |	32768		  | 7327     | 2673     | 73.269997000  | 
+----------------------+--------------------+---------------------+----------+----------+---------------+

The best combination that give us the best performance is :
LocalHistoryTableSize : 2048 and 4096 + GlobalPredictorSize = ChoicePredictorSize : 8192 with 74.84999800 % correctness.

*/

#include "Branch_Predictor.h"

const unsigned instShiftAmt = 2; // Number of bits to shift a PC by

// You can play around with these settings.
const unsigned localPredictorSize = 2048;
const unsigned localCounterBits = 2;
const unsigned localHistoryTableSize = 32768; 
const unsigned globalPredictorSize = 32768 ;
const unsigned globalCounterBits = 32;
const unsigned choicePredictorSize = 32768; // Keep this the same as globalPredictorSize.
const unsigned choiceCounterBits = 2;
const unsigned perceptronSize = 32768;

Branch_Predictor *initBranchPredictor()
{
    Branch_Predictor *branch_predictor = (Branch_Predictor *)malloc(sizeof(Branch_Predictor));

    #ifdef TWO_BIT_LOCAL
    branch_predictor->local_predictor_sets = localPredictorSize;
    assert(checkPowerofTwo(branch_predictor->local_predictor_sets));

    branch_predictor->index_mask = branch_predictor->local_predictor_sets - 1;

    // Initialize sat counters
    branch_predictor->local_counters =
        (Sat_Counter *)malloc(branch_predictor->local_predictor_sets * sizeof(Sat_Counter));

    int i = 0;
    for (i; i < branch_predictor->local_predictor_sets; i++)
    {
        initSatCounter(&(branch_predictor->local_counters[i]), localCounterBits);
    }
    #endif

    #ifdef TOURNAMENT
    assert(checkPowerofTwo(localPredictorSize));
    assert(checkPowerofTwo(localHistoryTableSize));
    assert(checkPowerofTwo(globalPredictorSize));
    assert(checkPowerofTwo(choicePredictorSize));
    assert(globalPredictorSize == choicePredictorSize);

    branch_predictor->local_predictor_size = localPredictorSize;
    branch_predictor->local_history_table_size = localHistoryTableSize;
    branch_predictor->global_predictor_size = globalPredictorSize;
    branch_predictor->choice_predictor_size = choicePredictorSize;
   
    // Initialize local counters 
    branch_predictor->local_counters =
        (Sat_Counter *)malloc(localPredictorSize * sizeof(Sat_Counter));

    int i = 0;
    for (i; i < localPredictorSize; i++)
    {
        initSatCounter(&(branch_predictor->local_counters[i]), localCounterBits);
    }

    branch_predictor->local_predictor_mask = localPredictorSize - 1;

    // Initialize local history table
    branch_predictor->local_history_table = 
        (unsigned *)malloc(localHistoryTableSize * sizeof(unsigned));

    for (i = 0; i < localHistoryTableSize; i++)
    {
        branch_predictor->local_history_table[i] = 0;
    }

    branch_predictor->local_history_table_mask = localHistoryTableSize - 1;

    // Initialize global counters
    branch_predictor->global_counters = 
        (Sat_Counter *)malloc(globalPredictorSize * sizeof(Sat_Counter));

    for (i = 0; i < globalPredictorSize; i++)
    {
        initSatCounter(&(branch_predictor->global_counters[i]), globalCounterBits);
    }

    branch_predictor->global_history_mask = globalPredictorSize - 1;

    // Initialize choice counters
    branch_predictor->choice_counters = 
        (Sat_Counter *)malloc(choicePredictorSize * sizeof(Sat_Counter));

    for (i = 0; i < choicePredictorSize; i++)
    {
        initSatCounter(&(branch_predictor->choice_counters[i]), choiceCounterBits);
    }

    branch_predictor->choice_history_mask = choicePredictorSize - 1;

    // global history register
    branch_predictor->global_history = 0;

    // We assume choice predictor size is always equal to global predictor size.
    branch_predictor->history_register_mask = choicePredictorSize - 1;
    #endif

    #ifdef GSHARE
    assert(checkPowerofTwo(globalPredictorSize));
    branch_predictor->global_predictor_size = globalPredictorSize;

    branch_predictor -> global_counters = (Sat_Counter *) malloc(globalPredictorSize * sizeof(Sat_Counter));
    for(int i=0; i < globalPredictorSize; i++)
   	    initSatCounter(&(branch_predictor->global_counters[i]),globalCounterBits);

    branch_predictor->global_history_mask = globalPredictorSize-1;

    branch_predictor->global_history =0;

    #endif

    #ifdef Perceptrons
    assert(checkPowerofTwo(globalPredictorSize));
    branch_predictor->global_predictor_size = globalPredictorSize;

    branch_predictor -> global_counters = (Sat_Counter *) malloc(globalPredictorSize * sizeof(Sat_Counter));
    for(int i=0; i < globalPredictorSize; i++)
        initSatCounter(&(branch_predictor->global_counters[i]),globalCounterBits);

    branch_predictor->global_history_mask = globalPredictorSize-1;
    branch_predictor->global_history =0;

    assert(checkPowerofTwo(perceptronSize));
    branch_predictor->perceptron_size = perceptronSize;
    branch_predictor->exact = 1.93 * globalCounterBits + 14; //suplement documetation exactly 0 = [1.93h + 14]

    unsigned perceptronBits = 1 + floor(log2(branch_predictor->exact));
    branch_predictor->perceptron_mask = perceptronSize - 1;
    branch_predictor->perceptron = (Perceptron *)malloc(perceptronSize * sizeof(Perceptron));
    for(int i = 0; i < perceptronSize; i++)
        initPercep(&(branch_predictor->perceptron[i]), globalCounterBits);
    
    #endif

    return branch_predictor ;

}

// sat counter functions
inline void initSatCounter(Sat_Counter *sat_counter, unsigned counter_bits)
{
    sat_counter->counter_bits = counter_bits;
    sat_counter->counter = 0;
    sat_counter->max_val = (1 << counter_bits) - 1;
}

inline void incrementCounter(Sat_Counter *sat_counter)
{
    if (sat_counter->counter < sat_counter->max_val)
    {
        ++sat_counter->counter;
    }
}

inline void decrementCounter(Sat_Counter *sat_counter)
{
    if (sat_counter->counter > 0) 
    {
        --sat_counter->counter;
    }
}

// Branch Predictor functions
bool predict(Branch_Predictor *branch_predictor, Instruction *instr)
{
    uint64_t branch_address = instr->PC;

    #ifdef TWO_BIT_LOCAL    
    // Step one, get prediction
    unsigned local_index = getIndex(branch_address, 
                                    branch_predictor->index_mask);

    bool prediction = getPrediction(&(branch_predictor->local_counters[local_index]));

    // Step two, update counter
    if (instr->taken)
    {
        // printf("Correct: %u -> ", branch_predictor->local_counters[local_index].counter);
        incrementCounter(&(branch_predictor->local_counters[local_index]));
        // printf("%u\n", branch_predictor->local_counters[local_index].counter);
    }
    else
    {
        // printf("Incorrect: %u -> ", branch_predictor->local_counters[local_index].counter);
        decrementCounter(&(branch_predictor->local_counters[local_index]));
        // printf("%u\n", branch_predictor->local_counters[local_index].counter);
    }

    return prediction == instr->taken;
    #endif

    #ifdef TOURNAMENT
    // Step one, get local prediction.
    unsigned local_history_table_idx = getIndex(branch_address,
                                           branch_predictor->local_history_table_mask);
    
    unsigned local_predictor_idx = 
        branch_predictor->local_history_table[local_history_table_idx] & 
        branch_predictor->local_predictor_mask;

    bool local_prediction = 
        getPrediction(&(branch_predictor->local_counters[local_predictor_idx]));

    // Step two, get global prediction.
    unsigned global_predictor_idx = 
        branch_predictor->global_history & branch_predictor->global_history_mask;

    bool global_prediction = 
        getPrediction(&(branch_predictor->global_counters[global_predictor_idx]));

    // Step three, get choice prediction.
    unsigned choice_predictor_idx = 
        branch_predictor->global_history & branch_predictor->choice_history_mask;

    bool choice_prediction = 
        getPrediction(&(branch_predictor->choice_counters[choice_predictor_idx]));


    // Step four, final prediction.
    bool final_prediction;
    if (choice_prediction)
    {
        final_prediction = global_prediction;
    }
    else
    {
        final_prediction = local_prediction;
    }

    bool prediction_correct = final_prediction == instr->taken;
    // Step five, update counters
    if (local_prediction != global_prediction)
    {
        if (local_prediction == instr->taken)
        {
            // Should be more favorable towards local predictor.
            decrementCounter(&(branch_predictor->choice_counters[choice_predictor_idx]));
        }
        else if (global_prediction == instr->taken)
        {
            // Should be more favorable towards global predictor.
            incrementCounter(&(branch_predictor->choice_counters[choice_predictor_idx]));
        }
    }

    if (instr->taken)
    {
        incrementCounter(&(branch_predictor->global_counters[global_predictor_idx]));
        incrementCounter(&(branch_predictor->local_counters[local_predictor_idx]));
    }
    else
    {
        decrementCounter(&(branch_predictor->global_counters[global_predictor_idx]));
        decrementCounter(&(branch_predictor->local_counters[local_predictor_idx]));
    }

    // Step six, update global history register
    branch_predictor->global_history = branch_predictor->global_history << 1 | instr->taken;
    branch_predictor->local_history_table[local_history_table_idx] = branch_predictor->local_history_table[local_history_table_idx] << 1 | instr->taken;
    // exit(0);
    //
    return prediction_correct;
    #endif

     #ifdef GSHARE
    //Design a XOR gate
    unsigned global_predictor_idx = ((branch_predictor->global_history ^ branch_address) & branch_predictor->global_history_mask);

    bool global_prediction = getPrediction(&(branch_predictor->global_counters[global_predictor_idx]));
    bool prediction_correct = global_prediction == instr -> taken;
    
    if (instr->taken){
	    incrementCounter(&(branch_predictor->global_counters[global_predictor_idx]));
	 
    }else{
	    decrementCounter(&(branch_predictor->global_counters[global_predictor_idx]));
    }

    branch_predictor->global_history = branch_predictor->global_history << 1 | instr->taken;

    return prediction_correct ;
    #endif

    #ifdef Perceptrons
    unsigned global_predictor_idx = ((branch_predictor->global_history ^ branch_address) & branch_predictor->global_history_mask);
    unsigned perceptron_idx = branch_predictor->perceptron_mask & branch_address;

    bool prediction_correct ;

    if (instr->taken){
        incrementCounter(&(branch_predictor->global_counters[global_predictor_idx]));
     
    }else{
        decrementCounter(&(branch_predictor->global_counters[global_predictor_idx]));
    }
    branch_predictor->global_history = branch_predictor->global_history << 1 | instr->taken;

    uint64_t theta = computePercep(&(branch_predictor->perceptron[perceptron_idx]), &(branch_predictor->global_counters[global_predictor_idx]));
    Perceptron_learning(&(branch_predictor->perceptron[perceptron_idx]), branch_predictor->exact, &(branch_predictor->global_counters[global_predictor_idx]), instr->taken, theta);
    while (theta>0){
        prediction_correct = theta == instr->taken;
    }

    return prediction_correct;
    #endif
}

void initPercep(Perceptron *perceptron, unsigned counter_bits)
{
    perceptron->coefficient = (uint64_t *)malloc(counter_bits * sizeof(uint64_t));
    for(int i = 0; i <= counter_bits; i++)
        perceptron->coefficient[i] = 1;    
        perceptron->perceptron_n = counter_bits;
}

uint64_t computePercep(Perceptron *perceptron, Sat_Counter *sat_counter)
{
    uint64_t theta = perceptron->coefficient[0];
    for(int i = 1; i <= sat_counter->counter_bits; i++){
        uint8_t temp = (sat_counter->counter & (1 << i));
        uint8_t h = 1;
        if(temp < 0)
            h = -1;
        theta += h * perceptron->coefficient[i];
    }
    return theta;
}

void Perceptron_learning(Perceptron *perceptron, unsigned exact, Sat_Counter *sat_counter, bool taken, uint64_t theta)
{
    if((theta < 0) == taken || (theta > 0) == taken || theta <= exact)
    for(int i = 0; i <= sat_counter->counter_bits; i++){
        int8_t temp = (sat_counter->counter & (1 << i));
        int8_t h = 1;
        int8_t x = 1;
        if(temp < 0)
            h = -1;
        if(taken == false)
            x = -1;
        perceptron->coefficient[i] = perceptron->coefficient[i] + (x*h);
    }
}

inline unsigned getIndex(uint64_t branch_addr, unsigned index_mask)
{
    return (branch_addr >> instShiftAmt) & index_mask;
}

inline bool getPrediction(Sat_Counter *sat_counter)
{
    uint8_t counter = sat_counter->counter;
    unsigned counter_bits = sat_counter->counter_bits;

    // MSB determins the direction
    return (counter >> (counter_bits - 1));
}

int checkPowerofTwo(unsigned x)
{
    //checks whether a number is zero or not
    if (x == 0)
    {
        return 0;
    }

    //true till x is not equal to 1
    while( x != 1)
    {
        //checks whether a number is divisible by 2
        if(x % 2 != 0)
        {
            return 0;
        }
        x /= 2;
    }
    return 1;
}



