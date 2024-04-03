import os.path

from matplotlib import pyplot as plt


def generate(model, noise):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(noise, training=False)

  return predictions


def save_images(predicted_image, epoch = None):


  fig = plt.figure(figsize=(10, 10))

  for i in range(predicted_image.shape[0]):
      plt.subplot(3, 3, i+1)
      plt.imshow((predicted_image[i, :, :, :].numpy() * 127.5 + 127.5).astype(int))
      plt.axis('off')
  plt.tight_layout()
  if not os.path.exists("./../catimages"):
      os.makedirs("./../catimages")
  plt.savefig('./../catimages/image_at_epoch_{:04d}.png'.format(epoch))
  plt.close(fig=fig)
