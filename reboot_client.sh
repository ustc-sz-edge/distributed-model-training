#!/usr/bin/expect
set host [lindex $argv 0]
set pass_wd [lindex $argv 1]
spawn ssh $host
expect "password:"
send "$pass_wd\n"
expect "*$"
send "sudo reboot\n"
expect "password:"
send "$pass_wd\n"
expect "*$"
