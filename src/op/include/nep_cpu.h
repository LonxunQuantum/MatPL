const int MAX_NEURON = 200; // maximum number of neurons in the hidden layer
const int MN = 1000;       // maximum number of neighbors for one atom
const int NUM_OF_ABC = 24;  // 3 + 5 + 7 + 9 for L_max = 4
const int MAX_NUM_N = 20;   // n_max+1 = 19+1
const int MAX_DIM = MAX_NUM_N * 7;
const int MAX_DIM_ANGULAR = MAX_NUM_N * 6;

const double C3B[NUM_OF_ABC] = {
  0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435, 0.596831036594608,
  0.596831036594608, 0.149207759148652, 0.149207759148652, 0.139260575205408, 0.104445431404056,
  0.104445431404056, 1.044454314040563, 1.044454314040563, 0.174075719006761, 0.174075719006761,
  0.011190581936149, 0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
  1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606};
const double C4B[5] = {
  -0.007499480826664, -0.134990654879954, 0.067495327439977, 0.404971964639861, -0.809943929279723};
const double C5B[3] = {0.026596810706114, 0.053193621412227, 0.026596810706114};
const double K_C_SP = 14.399645; // 1/(4*PI*epsilon_0)
// const double PI = 3.141592653589793;
// const double PI_HALF = 1.570796326794897;
const double PI = 3.14159265358979323846;
const double HALF_PI = 1.5707963267948966;
const int NUM_ELEMENTS = 103;

void cpu_dev_apply_mic(const double* box, double& x12, double& y12, double& z12);

void find_fc(double rc, double rcinv, double d12, double& fc);

void find_fn(const int n_max, const double rcinv, const double d12, const double fc12, double* fn);

void find_fn(const int n, const double rcinv, const double d12, const double fc12, double& fn);

void accumulate_s(const double d12, double x12, double y12, double z12, const double fn, double* s);

void find_q(const int n_max_angular_plus_1, const int n, const double* s, double* q);

void find_q_with_4body(const int n_max_angular_plus_1, const int n, const double* s, double* q);

void find_q_with_5body(const int n_max_angular_plus_1, const int n, const double* s, double* q);
