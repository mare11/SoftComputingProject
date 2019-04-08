import os

import video

# reading first lines
out = open('out.txt', 'r')
lines = out.readlines()
out.close()

# open again for writing and start with 2 same lines
out = open('out.txt', 'w')
out.writelines(lines[0:2])
path = 'data'
files = os.listdir(path)

for name in files:
    print(name)
    sum = video.load_video(os.path.join(path, name))
    out.write(name + '\t' + str(sum) + '\n')

out.close()
