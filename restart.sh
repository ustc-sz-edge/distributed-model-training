# gnome-terminal -x bash -c "./reboot_client.sh mzg@192.168.230.133 mzg;"
rank=("11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
ip=("11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
for i in 0 1 2 3 6
do
gnome-terminal -x bash -c "./reboot_client.sh edge${rank[$i]}@192.168.1.${ip[$i]} edge${rank[$i]};"
done

