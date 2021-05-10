from math import log
p1=4/9
p2=5/9
ent=-p1*log(p1,2)-p2*log(p2,2)
print(ent)

p1=3/4
p2=1/4
p3=1/5
p4=4/5
ent1=(-p1*log(p1,2)-p2*log(p2,2))*4/9
ent2=(-p3*log(p3,2)-p4*log(p4,2))*5/9
print(ent1+ent2)