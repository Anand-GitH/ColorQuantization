#####Clustering- Kmeans Algorithm and Color Quantization###########
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2

def k_means(k,data,iters,init):
    colors=[mcolors.TABLEAU_COLORS['tab:red'],mcolors.TABLEAU_COLORS['tab:green'],mcolors.TABLEAU_COLORS['tab:blue']]
    n_data=data.shape[0]
    n_feat=data.shape[1]
    init=init.astype('float64') 
    
    for i in range(iters):
        xdata=data
        
        #Calculate distance matrix and based on min distance assign clusters
        dist=(np.sum((data[None,:] - init[:, None])**2, -1)**0.5).T
        newclust=np.argmin(dist,axis=1)
        
        print("Classification Vector for iteration ",i+1,':',newclust)
        
        xdata=np.hstack((xdata,newclust.reshape(-1,1)))
        
        filename="task2_iter"+str(i+1)+"_a.jpg"
        plot_cluster(i+1,xdata,init,colors,filename)
        
        new_init = np.array([np.mean(data[newclust == i],axis=0) for i in range(k)])
        
        filename="task2_iter"+str(i+1)+"_b.jpg"
        plot_cluster(i+1,xdata,np.round(new_init,1),colors,filename)
        
        
        new_init=np.round(new_init,1)
        
        if np.all(init == new_init):
            break
        else:
            init = new_init
            
    return i+1,init,xdata
            
def k_means_color(k,data,iters,init):
    
    n_data=data.shape[0]
    n_feat=data.shape[1]
    init=init.astype('float64') 
    
    for i in range(iters):
    
        xdata=data
        #Calculate distance matrix and based on min distance assign clusters
        dist=(np.sum((data[None,:] - init[:, None])**2, -1)**0.5).T
        newclust=np.argmin(dist,axis=1)
    
        xdata=np.hstack((xdata,newclust.reshape(-1,1)))
        
        new_init = np.array([np.mean(data[newclust == i],axis=0) for i in range(k)])
        
        
        if np.all(init == new_init):
            break
        else:
            init = new_init
            
    return i+1,init,xdata

def plot_cluster(i,data,centers,colors,fname):
    n_feat=data.shape[1]
    k=centers.shape[0]
    
    plt.title('K-means Clustering Iteration:'+str(i))

    for c in range(k):
        
        tdata=np.array(data[data[:,-1] == c])[:,0:n_feat]    
        x=tdata[:,0]
        y=tdata[:,1]
        plt.scatter(x, y,s=100, marker='^',alpha=1, facecolors='none', edgecolors=colors[c])
        xc=centers[c,0]
        yc=centers[c,1]
        plt.scatter(xc,yc,s=100, marker='o', facecolors=colors[c], edgecolors=colors[c])
        plt.text(xc,yc,s='('+str(xc)+','+str(yc)+')',horizontalalignment='left',verticalalignment='top',size =8)
        
        
        for xs,ys in zip(x,y):
            plt.text(xs,ys,s='('+str(xs)+','+str(ys)+')',horizontalalignment='left',verticalalignment='top',size =8)
        
    plt.savefig(fname)
    plt.clf()
    plt.cla()
    plt.close()

def recreate_image(k,new_cent,xdata,original_shape):
    w, h, d=original_shape
    new_image = np.zeros((w * h, 3),dtype="float")
    
    for c in range(k):
        new_image[np.where(xdata[:,-1]==c),:]=new_cent[c,:]
     
    new_image=np.reshape(new_image,original_shape)
    new_image=new_image.astype("uint8")
    
    return new_image
    
def colorquantization(filename):
    
    print("Color Quantization using K-means clustering algorithm")
    print('################################################################################################')
    orig_img = cv2.imread(filename)
    nseed=123
    w, h, d = original_shape = tuple(orig_img.shape)
    inp_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB)
    image_array = np.reshape(inp_img, (w * h, 3))
    
    uniqcol=np.unique(image_array,axis=0)
    
    print("Number of unique colors used to represent current image:",len(uniqcol))
    
    for n_clust in ([3, 5, 10, 20]):
        
        ## Initialize centers for the clusters randomly from the image colors
        rng = np.random.RandomState(nseed)
        ridx= rng.permutation(uniqcol.shape[0])[:n_clust]
        init_cent = uniqcol[ridx]

        #Call k means to cluster colors in image
        iters,new_cent,xdata=k_means_color(k=n_clust,data=image_array,iters=50,init=init_cent)
        
        print("Number of colors: ",n_clust)
        q_image=recreate_image(k=n_clust,new_cent=new_cent,xdata=xdata,original_shape=original_shape)
   
        print('Quantized color values:',new_cent.astype("uint8"))
            
        #Convert back to color image
        q_image = cv2.cvtColor(q_image, cv2.COLOR_LAB2BGR)
        
        ifname="task2_baboon_"+str(n_clust)+".jpg"
        cv2.imwrite(ifname,q_image)
        
        print('################################################################################################')


#########Apply K-means for given dataset###########
#Testing K-means for the given data and see how it converges    
X=np.array([[5.9,3.2],
[4.6,2.9],
[6.2,2.8],
[4.7,3.2],
[5.5,4.2],
[5.0,3.0],
[4.9,3.1],
[6.7,3.1],
[5.1,3.8],
[6.0,3.0]])

init=np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])


iters,new_cent,xdata=k_means(k=3,data=X,iters=3,init=init)
print('###################################################################################')

##############################Color Quantization##################################
colorquantization("baboon.png")
print("Completed Task2")