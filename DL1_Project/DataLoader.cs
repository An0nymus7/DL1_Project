using System;
using System.Collections.Generic;
using Tensorflow;
using static DL1_Project.ImagePreprocessor;
using static DL1_Project.CocoAnnotation;
using Tensorflow.NumPy;
using System.IO;
using System.Drawing;
using System.Reflection.Metadata.Ecma335;
using System.Collections.Concurrent;
using System.Diagnostics;

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
        public (List<NDArray>, List<NDArray>, List<NDArray>, int) PrepareTrainingData()
        {
            int a = PreprocessBatch(trainImagesPath, trainImages, annotations, targetWidth, targetHeight, numClasses, 64, true); //maybe add bigger batch size
            return LoadSavedBatch(trainImages, true, 64, a);
        }

        public static void LoadSavedBatchBetweenEpoch(ref List<NDArray> imageBatches, ref List<NDArray> labelBatches, bool train, int startFrom)
        {
            const int MaxBatchLoad = 300; // Limit to 300 batches
            string directory = "../../../../../Data/ValidationBatch";
            if (train)
            {
                directory = "../../../../../Data/TrainBatch";
            }

            // Empty the lists to free memory
            imageBatches.Clear();
            labelBatches.Clear();
            GC.Collect();

            // Determine the total number of batches to load
            int totalBatches = Directory.GetFiles(directory, "*_images.npy").Length;
            int remainingBatches = Math.Min(MaxBatchLoad, totalBatches - startFrom);

            if (remainingBatches <= 0)
            {
                Console.WriteLine("No batches to load.");
                return;
            }

            var imageBatchBag = new ConcurrentBag<NDArray>();
            var labelBatchBag = new ConcurrentBag<NDArray>();

            Parallel.ForEach(Enumerable.Range(startFrom, remainingBatches), batchId =>
            {
                string batchImagePath = Path.Combine(directory, $"batch_{batchId}_images.npy");
                string batchLabelPath = Path.Combine(directory, $"batch_{batchId}_labels.npy");

                if (File.Exists(batchImagePath) && File.Exists(batchLabelPath))
                {
                    imageBatchBag.Add(np.load(batchImagePath));
                    labelBatchBag.Add(np.load(batchLabelPath));
                    Console.WriteLine($"Loaded batch {batchId} from disk.");
                    Debug.WriteLine($"Loaded batch {batchId} from disk.");
                }
            });

            // Transfer data back to lists after the parallel operation
            imageBatches.AddRange(imageBatchBag);
            labelBatches.AddRange(labelBatchBag);
        }

        private (List<NDArray>, List<NDArray>, List<NDArray>, int) LoadSavedBatch(List<CocoAnnotation.Image> images, bool knownSize, int batchSize, int batchCount)
        {
            string directory = "../../../../../Data/ValidationBatch";
            if (knownSize)
            {
                directory = "../../../../../Data/TrainBatch";
            }

            // Use thread-safe collections for parallel processing
            var imageBatchBag = new ConcurrentBag<NDArray>();
            var bboxBatchBag = new ConcurrentBag<NDArray>();
            var labelBatchBag = new ConcurrentBag<NDArray>();

            // Parallelize the batch loading
            Parallel.ForEach(Enumerable.Range(0, Math.Min(300, batchCount)), batchId =>
            {
                string batchImagePath = Path.Combine(directory, $"batch_{batchId}_images.npy");
                string batchBboxPath = Path.Combine(directory, $"batch_{batchId}_bboxes.npy");
                string batchLabelPath = Path.Combine(directory, $"batch_{batchId}_labels.npy");

                if (File.Exists(batchImagePath) && File.Exists(batchBboxPath) && File.Exists(batchLabelPath))
                {
                    // Load preprocessed batch from disk
                    var batchImages = np.load(batchImagePath);
                    var batchBboxes = np.load(batchBboxPath);
                    var batchLabels = np.load(batchLabelPath);

                    imageBatchBag.Add(batchImages);
                    bboxBatchBag.Add(batchBboxes);
                    labelBatchBag.Add(batchLabels);

                    Console.WriteLine($"Loaded batch {batchId} from disk.");
                }
            });

            // Transfer the thread-safe collections to lists
            var imageBatches = imageBatchBag.ToList();
            var bboxBatches = bboxBatchBag.ToList();
            var labelBatches = labelBatchBag.ToList();

            // Return lists of mini-batches
            return (imageBatches, bboxBatches, labelBatches, batchCount);
        }

        // Load the validation data by processing each image and annotation
        public (List<NDArray>, List<NDArray>, List<NDArray>, int) PrepareValidationData()
        {
            int a = PreprocessBatch(valImagesPath, valImages, annotations, targetWidth, targetHeight, numClasses, 32, false); //maybe add bigger batch size 
            return LoadSavedBatch(valImages, false, 32, a);
        }
    }
}
#region linear
//for (int i = startFrom; i < allbatch; i++)
//{
//    // Determine the batch ID and save paths
//    int batchId = i;
//    string batchImagePath = Path.Combine(directory, $"batch_{batchId}_images.npy");
//    //string batchBboxPath = Path.Combine(directory, $"batch_{batchId}_bboxes.npy");
//    string batchLabelPath = Path.Combine(directory, $"batch_{batchId}_labels.npy");

//    //if (File.Exists(batchImagePath) && File.Exists(batchBboxPath) && File.Exists(batchLabelPath))
//    if (File.Exists(batchImagePath) && File.Exists(batchLabelPath))
//    {
//        // Load preprocessed batch from disk
//        var batchImages = np.load(batchImagePath);
//        //var batchBboxes = np.load(batchBboxPath);
//        var batchLabels = np.load(batchLabelPath);

//        imageBatches.Add(batchImages);
//        //bboxBatches.Add(batchBboxes);
//        labelBatches.Add(batchLabels);

//        Console.WriteLine($"Loaded batch {batchId} from disk.");
//    }
//    else
//        break;
//}
#endregion