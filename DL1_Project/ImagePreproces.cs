using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;
using System.Drawing;
using Tensorflow;
using static Tensorflow.Binding;
using System.Drawing.Imaging;
using System.Reflection.Metadata.Ecma335;

namespace DL1_Project
{
    class ImagePreprocessor
    {
        //PreprocessImage: Resizes and normalizes a single image, and scales its bounding boxes.
        public static (NDArray, float[]) PreprocessImage(string ImagesPath, string imageName, List<float> bbox, int targetWidth, int targetHeight)
        {
            string fullPath = Path.Combine(ImagesPath, imageName);
            // Load the image using TensorFlow

            var decodedImage = tf.io.read_file(fullPath); // Load the image as a Tensor
            decodedImage = tf.image.decode_image(decodedImage); // Decode the JPEG

            // Resize the image
            var resizedImage = tf.image.resize(decodedImage, new int[] { targetHeight, targetWidth });

            // Normalize the image
            resizedImage = tf.divide(resizedImage, tf.constant(255.0f));

            // Scale bounding boxes
            float scaleX = (float)targetWidth / decodedImage.shape[1];
            float scaleY = (float)targetHeight / decodedImage.shape[0];
            float[] scaledBbox = new float[]
            {
                bbox[0] * scaleX,
                bbox[1] * scaleY,
                bbox[2] * scaleX,
                bbox[3] * scaleY
            };

            // Expand dimensions to match the batch format
            var batchedImage = tf.expand_dims(resizedImage, axis: 0);

            return (batchedImage.numpy(), scaledBbox);
        }

        //OneHotEncode: Converts the COCO category ID into a one-hot encoded vector.
        public static NDArray OneHotEncode(int categoryId, int numClasses)
        {
            NDArray oneHot = np.zeros(numClasses, np.float32);
            oneHot[categoryId-1] = 1.0f;
            return oneHot;
        }

        //PreprocessBatch: Preprocesses a batch of images and annotations, returning an array of images, bounding boxes, and labels ready for training.
        public static (List<NDArray>, List<NDArray>, List<NDArray>) PreprocessBatch(string ImagesPath, List<CocoAnnotation.Image> images, List<CocoAnnotation.Annotation> annotations, int targetWidth, int targetHeight, int numClasses, int batchSize, bool knownSize)
        {
            int actual_image_count = images.Count();
            string directory = "../../../../../Data/ValidationBatch";
            if (knownSize)
            {
                actual_image_count = 64000;
                directory = "../../../../../Data/TrainBatch";
            }


                SSDModel.UseGPU();
            // Create directory for saving progress
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Initialize lists to store mini-batches
            var imageBatches = new List<NDArray>();
            var bboxBatches = new List<NDArray>();
            var labelBatches = new List<NDArray>();

            // Process the dataset in chunks ||set a hard cap because it would go to the original size of 118287 and the file that contains the images is only 64000
            for (int i = 0; i < 1600; i += batchSize)
            {
                // Determine the batch ID and save paths
                int batchId = i / batchSize;
                string batchImagePath = Path.Combine(directory, $"batch_{batchId}_images.npy");
                string batchBboxPath = Path.Combine(directory, $"batch_{batchId}_bboxes.npy");
                string batchLabelPath = Path.Combine(directory, $"batch_{batchId}_labels.npy");

                if (File.Exists(batchImagePath) && File.Exists(batchBboxPath) && File.Exists(batchLabelPath))
                {
                    // Load preprocessed batch from disk
                    var batchImages = np.load(batchImagePath);
                    var batchBboxes = np.load(batchBboxPath);
                    var batchLabels = np.load(batchLabelPath);

                    imageBatches.Add(batchImages);
                    bboxBatches.Add(batchBboxes);
                    labelBatches.Add(batchLabels);

                    Console.WriteLine($"Loaded batch {batchId} from disk.");
                }
                else
                {
                    // Determine the actual batch size for the last mini-batch if it has fewer than batchSize items
                    int currentBatchSize = Math.Min(batchSize, images.Count - i);
                    //Console.WriteLine($"Batch Size: {currentBatchSize}, Target Width: {targetWidth}, Target Height: {targetHeight}");
                    //Console.WriteLine($"Total Elements: {currentBatchSize * targetWidth * targetHeight * 3}");

                    // Initialize NDArray placeholders for the current mini-batch
                    var batchImageArray = np.zeros(new Shape(currentBatchSize, targetWidth, targetHeight, 3), tf.float32);
                    var batchBboxArray = np.zeros(new Shape(currentBatchSize, 4), tf.float32);
                    var batchLabelArray = np.zeros(new Shape(currentBatchSize, numClasses), tf.float32);

                    
                    // Process mini-batch sequentially
                    for (int j = 0; j < currentBatchSize; j++)
                    {
                        var image = images[i + j];
                        var annotation = annotations[i + j];

                        // Preprocess the image and bounding box
                        if (File.Exists(Path.Combine(ImagesPath, image.file_name)))
                        {

                        (NDArray processedImage, float[] scaledBbox) = PreprocessImage(
                            ImagesPath,
                            image.file_name,
                            annotation.bbox,
                            targetWidth,
                            targetHeight
                        );

                        batchImageArray[j] = processedImage;
                        batchBboxArray[j] = scaledBbox;
                        batchLabelArray[j] = OneHotEncode(annotation.category_id, numClasses);
                        }
                        else
                            continue;
                    }

                    // Save processed batch to disk
                    np.save(batchImagePath, batchImageArray);
                    np.save(batchBboxPath, batchBboxArray);
                    np.save(batchLabelPath, batchLabelArray);

                    Console.WriteLine($"Saved batch {batchId} to disk.");

                    // Add mini-batch to the list of batches
                    imageBatches.Add(batchImageArray);
                    bboxBatches.Add(batchBboxArray);
                    labelBatches.Add(batchLabelArray);
                }
            }

            // Return lists of mini-batches
            return (imageBatches, bboxBatches, labelBatches);
        }
    }
}
