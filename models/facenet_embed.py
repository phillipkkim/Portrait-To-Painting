from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os  # Helps us find Phillip and Danny
import numpy as np
import torchvision.transforms as transforms


def get_embeddings(img):
    """img is a np array of the image"""
    # resnet = InceptionResnetV1(pretrained='vggface2').eval()
    # mtcnn = MTCNN()
    # img_cropped = mtcnn(img)
    # print(img_cropped)
    # img_embedding = resnet(img_cropped.unsqueeze(0))
    # return img_embedding

    """img is tensor of shape (1, 3, 256, 256)"""
    img = img.squeeze(0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    # temp = transforms.ToPILImage()(img)
    # temp = img.permute(0, 2, 3, 1).detach().numpy()
    # mtcnn = MTCNN()
    # img_cropped = mtcnn(temp)
    # if (img_cropped != None):
    #     img_embedding = resnet(img_cropped.unsqueeze(0))
    # else:
    img_embedding = resnet(img.unsqueeze(0))
    return img_embedding


if __name__ == '__main__':
    """ Load photos for testing """
    # Find path to Phillip and Danny
    #script_dir = os.path.dirname(__file__)
    # Absolute directory the script is in
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_file_path_Phillip = script_dir + "/Phillip.jpg"
    abs_file_path_Danny = script_dir + "/Danny.png"

    print(script_dir)

    # Upload our two test images
    # On successful execution of this statement,
    # an object of Image type is returned and stored in img variable)

    try:
        img1 = Image.open(abs_file_path_Phillip)
    except IOError:
        print("Unable to load Phillip photo")

    try:
        img2 = Image.open(abs_file_path_Danny)
    except IOError:
        print("Unable to load Danny photo")

    # If required, create a face detection pipeline using MTCNN:
    # mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)
    mtcnn = MTCNN()

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # img = Image.open(<image path>)

    # Get cropped and prewhitened image tensor
    # img_cropped = mtcnn(img, save_path=<optional save path>)
    img_cropped = mtcnn(img1, script_dir + "/image.jpg")
    img1 = np.array(img1)
    print(img1.shape)
    print(type(img_cropped))
    print(img_cropped.shape)
    img_new = mtcnn(img_cropped, script_dir + "/image2.jpg")

    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    print(img_embedding)

    # Or, if using for VGGFace2 classification
    # resnet.classify = True
    # img_probs = resnet(img_cropped.unsqueeze(0))
