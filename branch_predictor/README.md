Project Branch_Predictor

Part 1: TWO_BIT_LOCAL Predictor

LocalPredictorSize 	LocalCounterBits	Correct	Incorrect	% Correctness

2048	1	8217	1783	82.169998000
2048	2	8467	1533	84.669998000
4096	2	8468	1532	84.680000000
8192	2	8476	1524	84.759995000
16384	2	8474	1526	84.739998000
32768	2	8473	1527	84.729996000
65536	2	8473	1527	84.729996000


The best combination that give us the best performance is LocalPredictorSize : 8192 and LocalCounterBits: 2 with 84.759995% correctness.


Part 2: TOURNAMENT Predictor:


LocalHistoryTableSize	GlobalPredictorSize	ChoicePredictorSize	Correct	Incorrect	% Correctness
2048	8192	8192	8060	1940	80.599998000
4096	8192	8192	8072	1928	80.720001000
4096	16384	16384	8064	1936	80.639999000
8192	16384	16384	8064	1936	80.639999000
16384	32768	32768	8046	1954	80.459999000


The best combination that give us the best performance is LocalHistoryTableSize : 4096 and GlobalPredictorSize = ChoicePredictorSize : 8192 with 80.720001% correctness.


Part 3: GSHARE Predictor:

LocalHistoryTableSize	GlobalPredictorSize	ChoicePredictorSize	Correct	Incorrect	% Correctness

2048	8192	8192	7485	2215	74.84999800
4096	8192	8192	7485	2215	74.84999800
4096	16384	16384	7422	2578	74.22000100
8192	16384	16384	7422	2578	74.22000100
16384	32768	32768	7327	2673	73.26999700


The best combination that give us the best performance is LocalHistoryTableSize :
2048 and 4096 + GlobalPredictorSize = ChoicePredictorSize : 8192 with 74.84999800 % correctness.



