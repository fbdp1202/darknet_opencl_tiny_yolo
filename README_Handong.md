안녕하세요 한동대 가속기 팀입니다.
저희는 Tiny-yolo v2 코드를 OpenCL로 가속시켰고,
이를 위해서 몇몇 부분 코드 수정을 하였습니다.

실행방법은 다음과 같습니다
user:~$ ./run.sh
위 'run.sh' bashshell 파일로 실행하면 됩니다.

OPENCL를 사용하지 않고 CPU만 사용하기 위해서는
Makefile에서 "OPENCL=1" 옵션을 "OPENCL=0"으로 하면 됩니다.

이 버전은 Half와 Vectorization이 사용되었습니다.
Half를 사용하기 위해서 'define_cl.h' Line 10에
pragma가 기술되어 있습니다.

Vectorization는 LEN_VEC {1,2,4,8,16} 이 구현되어 있으며
Tiny Yolo에 12,13번째 Convolutional layer에 적용 시켰습니다.

저희는 Maximum WorkGroup Item이 169(13*13)보다 크다는 전제로
 코드를 작성하였습니다.

OPENCL로 작성한 Kernel Code는 'C_block_form_yechan.cl'
 파일 입니다.
Convolutional Layer와 같은 경우 실행되는 함수는
if LEN_VEC = 1, function 'Conv3'
if LEN_VEC = 2, function 'Conv3_vec'
if LEN_VEC = 4, function 'Conv3_vec4'
if LEN_VEC = 8, function 'Conv3_vec8'
if LEN_VEC = 16, function 'Conv3_vec18'

Maxpooling과 같은 경우 실행되는 함수는
if Stride = 2, function 'Pool'
if Stride = 1, function 'Pool2'

********************** Half 사용법 ************************
저희가 Half를 사용하기 위해서 Host에서는  함수  'do_conversion
_f_to_h'를 구현하여 Float를 Half로 변경한 뒤에 Global Memory
로 전송시켰습니다. 이 뒤에 OpenCL에서 제공하는 함수인 
'vload_half'와 'vstore_half'를 이용하여 Half형태를 float로 
Load, float를 Half로 Store 진행했습니다.
**********************************************************

설명이 부족하거나 궁금한 점이 있으시다면,
"21500429@handong.edu"로 연락주시기 바랍니다. 감사합니다.
