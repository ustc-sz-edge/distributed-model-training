rank=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10")
nine_rank=("01" "02" "03" "04" "05" "05" "06" "07" "08" "09")
user_name=("11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
ip=("11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
client_port=("47001" "47002" "47003" "47004" "47005" "47006" "47007" "47008" "47009" "47010")
master_port=("57001" "57002" "57003" "57004" "57005" "57006" "57007" "57008" "57009" "57010")
T_B=1500
dir='./res/'
test_batch_size=32
world_size=9
type='c10'
mode='adacons'
bs=16
ada_lr=0.01
cons_lr=0.003
GPU=2
c=5.0
m_min=4
m_max=64

# for i in 0 1 2 3 5 6 7 8 9
for i in 0 1 3 6
do
gnome-terminal -x bash -c "./run_client.sh edge${user_name[$i]}@192.168.1.${ip[$i]} edge${user_name[$i]} 'cd /data/lwang/client_module/;python3 client.py --listen_port ${client_port[$i]} --master_listen_port ${master_port[$i]} --idx $i';exec bash;"
done
gnome-terminal -x bash -c "source /etc/profile;cd /data/lwang/distirbuted-model-training;/opt/anaconda3/envs/pytorch/bin/python3 /data/lwang/distirbuted-model-training/server.py;exec bash;"
