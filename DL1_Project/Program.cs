﻿using System.IO;
using Newtonsoft.Json;
using Tensorflow;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace DL1_Project
{
    class Program
    {
        static CocoAnnotation.Root LoadCocoAnnotations(string path)
        {
            var json = File.ReadAllText(path);
            return JsonConvert.DeserializeObject<CocoAnnotation.Root>(json);
        }

        static void Main(string[] args)
        {
            var train_annotations = LoadCocoAnnotations("../../../../../Data/instances_train2017.json");
            var val_annotations = LoadCocoAnnotations("../../../../../Data/instances_val2017.json");
            string traing_Image_path = "../../../../../Data/train2017";
            string val_Image_path = "../../../../../Data/val2017";


            var dataloader = new DataLoader(traing_Image_path, val_Image_path, train_annotations.images, val_annotations.images, train_annotations.annotations, 300, 300, train_annotations.categories.Count);


            RunTraining(dataloader, 300, 300, train_annotations.categories.Count);

            Console.WriteLine($"Loaded {train_annotations.images.Count} images and {train_annotations.annotations.Count} annotations.");
            Console.WriteLine($"Loaded {val_annotations.images.Count} images and {val_annotations.annotations.Count} annotations.");
        }


        public static void RunTraining(DataLoader dataLoader, int inputHeight, int inputWidth, int numClasses)
        {
            //Build the SSD model
            var model = SSDModel.BuildSSDModel(inputHeight, inputWidth, numClasses);

            //Prepare training and validation data
            var (trainImages, trainBboxes, trainLabels) = dataLoader.PrepareTrainingData();
            var (valImages, valBboxes, valLabels) = dataLoader.PrepareValidationData();

            //Train the model
            Console.WriteLine("Starting training...");
            SSDModel.TrainModel(model, trainImages, trainBboxes, trainLabels,10);

            //Evaluate the model
            Console.WriteLine("Evaluating the model...");
            SSDModel.EvaluateModel(model, valImages, valBboxes, valLabels);
        }

    }
}
