import imageio
ii=0
images=[]
while ii<341:
    images.append(imageio.imread("./images/Data{:}.png".format(ii)))
    ii+=1
imageio.mimsave('./movie.gif', images)
