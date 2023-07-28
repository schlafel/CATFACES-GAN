import imageio.v2 as imageio
import os


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont



filenames = os.listdir("catimages")
full_dir = os.path.basename("catimages")

images = []
for filename in filenames:
    img = Image.open(os.path.join(full_dir,filename))

    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(img)

    # Custom font style and font size
    myFont = ImageFont.truetype('arial.ttf', 65)

    # Add Text to an image
    I1.text((10, 10),
            "Epoch: {:d}".format(int(filename.split("_")[-1].replace(".png",""))),
            font=myFont, fill=(255, 0, 0))

    # Display edited image
    #img.show()

    images.append(img)
    #imageio.imread(os.path.join(full_dir,filename))
imageio.mimsave('DCGAN_CATFACES.gif', images)