#!/usr/bin/env python
import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageFile
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

class App:
    def __init__(self):
        # define all classes available for this classifier
        self.classes = ['Affenpinscher', 'Afghan_hound', 'Airedale_terrier', 'Akita', 'Alaskan_malamute', 'American_eskimo_dog', 'American_foxhound',
         'American_staffordshire_terrier', 'American_water_spaniel', 'Anatolian_shepherd_dog', 'Australian_cattle_dog', 'Australian_shepherd',
         'Australian_terrier', 'Basenji', 'Basset_hound', 'Beagle', 'Bearded_collie', 'Beauceron', 'Bedlington_terrier', 'Belgian_malinois',
         'Belgian_sheepdog', 'Belgian_tervuren', 'Bernese_mountain_dog', 'Bichon_frise', 'Black_and_tan_coonhound', 'Black_russian_terrier', 'Bloodhound',
         'Bluetick_coonhound', 'Border_collie', 'Border_terrier', 'Borzoi', 'Boston_terrier', 'Bouvier_des_flandres', 'Boxer', 'Boykin_spaniel', 'Briard',
         'Brittany', 'Brussels_griffon', 'Bull_terrier', 'Bulldog', 'Bullmastiff', 'Cairn_terrier', 'Canaan_dog', 'Cane_corso', 'Cardigan_welsh_corgi',
         'Cavalier_king_charles_spaniel', 'Chesapeake_bay_retriever', 'Chihuahua', 'Chinese_crested', 'Chinese_shar-pei', 'Chow_chow', 'Clumber_spaniel',
         'Cocker_spaniel', 'Collie', 'Curly-coated_retriever', 'Dachshund', 'Dalmatian', 'Dandie_dinmont_terrier', 'Doberman_pinscher',
         'Dogue_de_bordeaux', 'English_cocker_spaniel', 'English_setter', 'English_springer_spaniel', 'English_toy_spaniel', 'Entlebucher_mountain_dog',
         'Field_spaniel', 'Finnish_spitz', 'Flat-coated_retriever', 'French_bulldog', 'German_pinscher', 'German_shepherd_dog',
         'German_shorthaired_pointer', 'German_wirehaired_pointer', 'Giant_schnauzer', 'Glen_of_imaal_terrier', 'Golden_retriever', 'Gordon_setter',
         'Great_dane', 'Great_pyrenees', 'Greater_swiss_mountain_dog', 'Greyhound', 'Havanese', 'Ibizan_hound', 'Icelandic_sheepdog',
         'Irish_red_and_white_setter', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_chin',
         'Keeshond', 'Kerry_blue_terrier', 'Komondor', 'Kuvasz', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberger', 'Lhasa_apso', 'Lowchen',
         'Maltese', 'Manchester_terrier', 'Mastiff', 'Miniature_schnauzer', 'Neapolitan_mastiff', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_buhund',
         'Norwegian_elkhound', 'Norwegian_lundehund', 'Norwich_terrier', 'Nova_scotia_duck_tolling_retriever', 'Old_english_sheepdog', 'Otterhound',
         'Papillon', 'Parson_russell_terrier', 'Pekingese', 'Pembroke_welsh_corgi', 'Petit_basset_griffon_vendeen', 'Pharaoh_hound', 'Plott', 'Pointer',
         'Pomeranian', 'Poodle', 'Portuguese_water_dog', 'Saint_bernard', 'Silky_terrier', 'Smooth_fox_terrier', 'Tibetan_mastiff',
         'Welsh_springer_spaniel', 'Wirehaired_pointing_griffon', 'Xoloitzcuintli', 'Yorkshire_terrier']

        # define in which device it will run
        self.device = "cpu"

    def predict(self, img_path):
        '''
        Predicts if the image file inform has a dog or human or neither.
        Args:
            img_path: image to be classify
        Return:
        '''
        print(">>> start prediction")
        # transform the image file to tensor
        image_normalized = self.process_image(img_path, true)
        image_not_normalized = self.process_image(img_path, false)
        print(">>> image processed")

        is_dog = self.is_dog(image_normalized)
        print(">>> does image have a dog?", is_dog)
        is_human = self.is_human(img_path)
        print(">>> does image have a human face?", is_human)
        dogs_breed = None

        if is_dog or is_human:
            # get the dog's breed by the class found
            dogs_breed = self.get_dogs_breed(image_not_normalized)
            print(">>> getting dog's breed ... and", dogs_breed)

        # build and return the msg base on the variables
        # TODO add a folder with one picture of each dog and show side by side with a human
        message = self.build_message(is_dog, is_human, dogs_breed)
        print(">>> building message -> ", message)
        return message

    def process_image(self, img_path, normalize):
        '''
        Process the image file, resizing and normalized it
        Args:
            img_path: image path to be processed
        Return:
            Returns a image as a tensor
        '''
        print(">>> processing image")
        # load the image
        image = Image.open(img_path)
        # converting to RGB fixes the png problem with transparency
        image = image.convert('RGB')
        # resize to max of 680px
        width, height = image.size[:2]
        if height > width:
            baseheight = 680
            hpercent = (baseheight/float(image.size[1]))
            wsize = int((float(image.size[0])*float(hpercent)))
            image = image.resize((wsize, baseheight), Image.ANTIALIAS)
            image.save(img_path)
        else:
            basewidth = 680
            wpercent = (basewidth/float(image.size[0]))
            hsize = int((float(image.size[1])*float(wpercent)))
            image = image.resize((basewidth,hsize), Image.ANTIALIAS)
            image.save(img_path)
        # resize, normlize and transform into tensor
        if (normalize):
            transform_pipeline = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_pipeline = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224)
                transforms.ToTensor()
            ])

        # preprocessing using a transform pipeline.
        image = transform_pipeline(image)
        # PyTorch pretrained models expect the Tensor dims to be (num input imgs, num color channels, height, width).
        # Currently however, we have (num color channels, height, width); let's fix this by inserting a new axis.
        image = image.unsqueeze(0)  # Insert the new axis at index 0 i.e. in front of the other axes/dims.
        # set device to process in (cuda if gpu available otherwie, cpu)
        return image.to(self.device)

    def is_human(self, img_path):
        '''
        Identifies if it has human face on the image using opencv haar cascades which has 98% accuracy.
        Args:
            img_path: image path to be identified
        Return:
            True or False if the image contains a human face
        '''
        face_cascade = cv2.CascadeClassifier(os.path.abspath('./resources/haarcascades/haarcascade_frontalface_alt.xml'))
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def is_dog(self, image):
        '''
        Identifies if it has dog on the image using resnet50 which has 99% accuracy.
        Args:
            image: image tensor to be identified
        Return:
            True or False if the image contains a dog
        '''
        # predict dogs using the resnet50
        resnet50 = models.resnet50(pretrained=True).to(self.device)
        resnet50 = resnet50.eval()
        resnet50 = resnet50(Variable(image))
        # get the max index
        class_index = resnet50.detach().cpu().numpy().argmax()
        print(">>> restnet50 found", class_index, "index")
        # indexes between 151 and 268 on imagenet means dogs
        return 151 <= class_index <= 268  # true/false

    def build_message(self, is_dog, is_human, dogs_breed):
        '''
        Build a msg to show to the user
        Args:
            is_dog: it informs if it has a dog on the image
            is_human: it informs if it has a human on the picture
            dogs_breed: in case of a dog or human, it will have the dog's breed
        Return:
            Returns an appropriate message base on the parameters to be show on the web
        '''
        if is_dog:
            return "Woof Woof... it is a dog! <br/> Predicted breed: {breed}".format(breed=dogs_breed)
        elif is_human:
            return "Hello Human!<br/>If you were a dog you could look like a {breed}".format(breed=dogs_breed)
        else:
            return "Woow ,You are neither human, nor dog!"

    def get_model(self):
        '''
        Loads and prepares the model for dog's breed classifier.
        Return:
            model which classifies dog's breed
        '''
        # get base network
        model = models.vgg16(pretrained=False)
        
        for param in model_transfer.parameters():
            param.requires_grad = False
            
        # Define dog breed classifier
        model_transfer.classifier = nn.Sequential(
            nn.Linear(25088, 4096), nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(4096, 512), 
            nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(512, 133))

        # load trained model from disk
        model.load_state_dict(torch.load(os.path.abspath('./resources/model/model_transfer.pt'), map_location=torch.device('cpu')))
       
        # set model to evaluation
        model.eval()
        return model

    def get_dogs_breed(self, image):
        '''
        It returns the dog's breed.
        Args:
            image: received a image (tensor) already transformed to be classified
        Return:
            Returns the dog's breed name
        '''
        # load the trained model for evaluation
        model = self.get_model().to(self.device)
        # predict
        prediction = model(Variable(image))
        # get the max index
        class_index = prediction.detach().cpu().numpy().argmax()
        # get the dog's breed by the class found
        return self.classes[class_index].replace('_', ' ')
