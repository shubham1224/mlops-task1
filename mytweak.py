import os
import sys
import fileinput


def main(args):
 print(args[0])

 fin = open("mymodel.py", "rt")

 data = fin.read()

 data = data.replace('architecture'+str(int(args[0])-1),'architecture'+args[0])

 fin.close()

 fin = open("mymodel.py", "wt")
 
 fin.write(data)

 fin.close()

if __name__=='__main__':
 main(sys.argv[1:])



































#def main(args):
# print(args[0])
#
# reading_file = open("mymodel.py", "r")
#
# new_file_content = ""
# for line in reading_file:
#   stripped_line = line.strip()
 # new_line = stripped_line.replace("model=mymodule.architecture1(model)", "model=mymodule.architecture2(model)")

#   new_line = stripped_line.replace("architecture"+str(int(args[0])-1), "architecture"+args[0])
#   new_file_content += new_line +"\n"
# reading_file.close()

# writing_file = open("mymodel.py", "w")
# writing_file.write(new_file_content)
# writing_file.close()

#if __name__=='__main__':
# main(sys.argv[1:])

