from itertools import permutations

part_1= """
subnets: [1,4]
topology: [[ 1, 1, 0],
           [ 1, 1, 1],
           [ 0, 1, 1]]
sensitive_hosts:
"""
part_2="""
os:
  - Rasbian
  - Penbox
services:
  - samba3.0.20
  - samba4.1
  - opensmtp6.0.6
  - opensmtp8
  - vsftpd-2.3.4
  - vsftpd-6.0.0
  - proftpd-5
  - proftpd-1.3.3d
  - xrdp
  - mysql
  - openssh
  - ApacheJserver
  - init
  - http
vul:
  - CVE20072447
  - CVE20201938

processes:
  - shadow

exploits:
  e_init:
    service: init
    os: None
    vul : None
    prob: 1.0
    cost: 1
    access: root
    credentials_needed: 0

  e_ssh_cred1_login:
    service: openssh
    os: None
    vul: None
    prob: 1.0
    cost: 3
    access: root
    credentials_needed: 1

  e_ssh_cred2_login:
    service: openssh
    os: None
    vul: None
    prob: 1.0
    cost: 3
    access: root
    credentials_needed: 2

  e_ssh_cred3_login:
    service: openssh
    os: None
    vul: None
    prob: 1.0
    cost: 3
    access: root
    credentials_needed: 3

  e_samba3.0.20:
    service: samba3.0.20
    os: None
    vul: CVE20072447
    prob: 1.0
    cost: 3
    access: root
    credentials_needed: 0

  e_opensmtp6.0.6:
    service: opensmtp6.0.6
    os: None
    vul: None
    prob: 1.0
    cost: 3
    access: root
    credentials_needed: 0

  e_vsftpd-2.3.4:
    service: vsftpd-2.3.4
    os: None
    vul: None
    prob: 1.0
    cost: 3
    access: root
    credentials_needed: 0

  e_proftpd-1.3.3d:
    service: proftpd-1.3.3d
    os: None
    vul: None
    prob: 1.0
    cost: 3
    access: root
    credentials_needed: 0


  e_ApacheJserver:
    service: ApacheJserver
    os: None
    vul: CVE20201938
    prob: 1.0
    cost: 3
    access: root
    credentials_needed: 0

privilege_escalation:
  pe_gain_Credentails_from_Shadow:
    process: shadow
    os: Rasbian
    prob: 1.0
    cost: 3
    access: root
    credentials_tofind: 1

service_scan_cost: 1
os_scan_cost: 1
vul_scan_cost: 1
subnet_scan_cost: 1
process_scan_cost: 1
wiretapping_cost: 1

host_configurations:
  (1, 0):
    os: Penbox
    services: [ init ]
    vul: []
    processes: [ ]
    credentials_needed: 0
    credentials_tofind: 0
"""
part_3 = """
firewall:
  (0, 1): [samba3.0.20, http, samba4.1, opensmtp6.0.6, opensmtp8, vsftpd-2.3.4, vsftpd-6.0.0, proftpd-5, proftpd-1.3.3d, xrdp, mysql, openssh, ApacheJserver, init]
  (1, 0): [samba3.0.20, http, samba4.1, opensmtp6.0.6, opensmtp8, vsftpd-2.3.4, vsftpd-6.0.0, proftpd-5, proftpd-1.3.3d, xrdp, mysql, openssh, ApacheJserver, init]
  (2, 1): [samba3.0.20, http, samba4.1, opensmtp6.0.6, opensmtp8, vsftpd-2.3.4, vsftpd-6.0.0, proftpd-5, proftpd-1.3.3d, xrdp, mysql, openssh, ApacheJserver, init]
  (1, 2): [samba3.0.20, http, samba4.1, opensmtp6.0.6, opensmtp8, vsftpd-2.3.4, vsftpd-6.0.0, proftpd-5, proftpd-1.3.3d, xrdp, mysql, openssh, ApacheJserver, init]
step_limit: 100
"""

host_0 ="""
    os: Rasbian
    services: [ openssh,samba3.0.20 ]
    vul: [ ]
    processes: [ ]
    credentials_needed: 0
    credentials_tofind: 0
"""

host_1 ="""
    os: Rasbian
    services: [ openssh,samba3.0.20 ] #samba3.0.20 to hide
    vul: [ CVE20072447 ]
    processes: [ ]
    credentials_needed: 0
    credentials_tofind: 0
"""

host_2 ="""
    os: Rasbian
    services: [openssh,http,opensmtp6.0.6]
    vul: []
    processes: []
    credentials_needed: 0
    credentials_tofind: 0
"""

host_3 ="""
    os: Rasbian
    services: [ proftpd-1.3.3d,openssh ]
    vul: [ ]
    processes: [ ]
    credentials_needed: 0
    credentials_tofind: 0
"""

switch = {
    0: host_0,
    1: host_1,
    2: host_2,
    3: host_3
}

if __name__ == "__main__":
    per = permutations([0, 1, 2, 3])
    for idx, elem in enumerate(per):
        with open("s1_r_a_"+str(idx)+".yaml", "a") as file:
            file.write(part_1)
            for idx2, elem1 in enumerate(elem):
                if elem1 != 0:
                    file.write("  (2, " + str(idx2) + "): 100 \n")
            file.write(part_2)
            for idx2, elem1 in enumerate(elem):
                file.write("  (2, "+ str(idx2) +"):")
                file.write(switch.get(elem1))
            file.write(part_3)
