using System;
using System.Collections.Generic;
using Tensorflow;
using static DL1_Project.ImagePreprocessor;
using static DL1_Project.CocoAnnotation;
using Tensorflow.NumPy;

namespace DL1_Project
{
    class DataLoader
    {
        private readonly string trainImagesPath;
        private readonly string valImagesPath;
        private readonly List<CocoAnnotation.Image> trainImages;
        private readonly List<CocoAnnotation.Image> valImages;
        private readonly List<CocoAnnotation.Annotation> annotations;
        private readonly int targetWidth;
        private readonly int targetHeight;
        private readonly int numClasses;

        public DataLoader(string trainImagesPath, string valImagesPath, List<CocoAnnotation.Image> trainImages,
                          List<CocoAnnotation.Image> valImages, List<CocoAnnotation.Annotation> annotations,
                          int targetWidth, int targetHeight, int numClasses)
        {
            this.trainImagesPath = trainImagesPath;
            this.valImagesPath = valImagesPath;
            this.trainImages = trainImages;
            this.valImages = valImages;
            this.annotations = annotations;
            this.targetWidth = targetWidth;
            this.targetHeight = targetHeight;
            this.numClasses = numClasses;
        }

        // Load the training data by processing each image and annotation
        public (List<NDArray> , List<NDArray> , List<NDArray> ) PrepareTrainingData()
        {
            return PreprocessBatch(trainImagesPath,trainImages, annotations, targetWidth, targetHeight, numClasses,32);
        }

        // Load the validation data by processing each image and annotation
        public (List<NDArray>, List<NDArray> , List<NDArray> ) PrepareValidationData()
        {
            return PreprocessBatch(valImagesPath,valImages, annotations, targetWidth, targetHeight, numClasses,32);
        }
    }
}
