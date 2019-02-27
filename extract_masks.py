import numpy as np

def color_loc(bb2):
    #bb2 = plt.imread(img_loc)
    blue_locs = np.array(np.where(np.logical_and(bb2[:,:,0]==0,bb2[:,:,1]==0))).T
    green_locs = np.array(np.where(np.logical_and(bb2[:,:,0]==0,bb2[:,:,1]==1))).T
    yellow_locs = np.array(np.where(np.logical_and(bb2[:,:,0]==1,bb2[:,:,1]==1))).T
    pink_locs = np.array(np.where(np.logical_and(bb2[:,:,0]==1,bb2[:,:,1]==0))).T
    white_locs = np.array(np.where(bb2[:,:,0]+bb2[:,:,1]+bb2[:,:,2]+bb2[:,:,3]==0)).T
    
    blue_set = set(tuple(i) for i in blue_locs)
    green_set = set(tuple(i) for i in green_locs)
    yellow_set = set(tuple(i) for i in yellow_locs)
    pink_set = set(tuple(i) for i in pink_locs)
    white_set = set(tuple(i) for i in white_locs)

    blue = blue_set - white_set
    green = green_set - white_set
    yellow = yellow_set - white_set
    pink = pink_set - white_set
    return blue,green,yellow,pink 


if __name__ == "__main__":
    r_folder = 'review1/'
    for main_path,_,k in os.walk('result1'):
        if 'masks' in main_path and isinstance(k,list) and len(k)>0:
            #print(main_path)
    #         print(k)
            name = main_path.split('\\')[1]
            ser = main_path.split('\\')[2]
            #print(name)
            #break
            #break
            if not os.path.isdir(r_folder+name+'/'+ser):
                os.makedirs(r_folder+name+'/'+ser+'/Aorta')
                os.makedirs(r_folder+name+'/'+ser+'/lvo')
                os.makedirs(r_folder+name+'/'+ser+'/rostium')
                os.makedirs(r_folder+name+'/'+ser+'/lostium')

            for mask_name in k:

                #mask_name = 'Mask - 157.png'
                mask_img = plt.imread(main_path+'/'+mask_name)
                mask_img[mask_img>0]=1.0
                #print(mask_name)
                true_img = np.array(Image.open(main_path.replace('masks','images')+'/'+mask_name.replace('Mask','img')))

                true_img2 = np.zeros((true_img.shape[0],true_img.shape[1],3))

                true_img2[:,:,0] = true_img
                true_img2[:,:,1] = true_img
                true_img2[:,:,2] = true_img

                true_img = true_img2

                scipy.misc.imsave('temp1.png',true_img)


                true_img = np.array(Image.open('temp1.png'))
                #plt.imshow(true_img),plt.show()
                #plt.imread(i.replace('masks','images')+'/'+mask.replace('Mask','img')).astype(np.float)

                b,g,y,p = color_loc(mask_img)