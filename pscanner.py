import socket

srv_addr = input("enter ip :")
srv_port = input("enter port range :")

lport = int(srv_port.split("-")[0])
hport = int(srv_port.split("-")[1])

print("searching for ports on ", srv_addr, "in range ", srv_port)

for port in range(lport,hport):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  s.settimeout(0.5)
  status = s.connect_ex((srv_addr,port))
  if(status == 0):
    print("*** Port",port, " - OPEN ***")
  else:
    print("Port ", port, "closed")
  s.close()
