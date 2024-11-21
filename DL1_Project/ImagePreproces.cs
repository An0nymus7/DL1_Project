using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;
using System.Drawing;
using Tensorflow;

namespace DL1_Project
{
    class ImagePreprocessor
    {
        //PreprocessImage: Resizes and normalizes a single image, and scales its bounding boxes.
        public static (NDArray, float[]) PreprocessImage(string ImagesPath,string imageName, List<float> bbox, int targetWidth, int targetHeight)
        {
            string fullPath = Path.Combine(ImagesPath,imageName);
            //kép betöltés
            Bitmap bitmap = new Bitmap(fullPath);
            bitmap = new Bitmap(bitmap, new Size(targetWidth, targetHeight));

            //NDArray konvertálás
            NDArray npArray = np.zeros(new Shape(1, targetWidth, targetHeight, 3), np.float32);
            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    Color color = bitmap.GetPixel(x, y);
                    npArray[0, y, x, 0] = color.R / 255.0f;
                    npArray[0, y, x, 1] = color.G / 255.0f;
                    npArray[0, y, x, 2] = color.B / 255.0f;
                }
            }

            float scaleX = (float)targetWidth / bitmap.Width;
            float scaleY = (float)targetHeight / bitmap.Height;
            float[] scaledBbox = new float[]
                {
                    bbox[0] * scaleX,
                    bbox[1] * scaleY,
                    bbox[2] * scaleX,
                    bbox[3] * scaleY
                };
            return (npArray, scaledBbox);
        }

        //OneHotEncode: Converts the COCO category ID into a one-hot encoded vector.
        public static NDArray OneHotEncode(int categoryId, int numClasses)
        {
            NDArray oneHot = np.zeros(numClasses, np.float32);
            oneHot[categoryId] = 1.0f;
            return oneHot;
        }

        //PreprocessBatch: Preprocesses a batch of images and annotations, returning an array of images, bounding boxes, and labels ready for training.
        public static (List<NDArray> , List<NDArray> , List<NDArray> ) PreprocessBatch(string ImagesPath,List<CocoAnnotation.Image> images, List<CocoAnnotation.Annotation> annotations, int targetWidth, int targetHeight, int numClasses, int batchSize)
        {
            // Initialize lists to store mini-batches
            var imageBatches = new List<NDArray>();
            var bboxBatches = new List<NDArray>();
            var labelBatches = new List<NDArray>();

            // Process the dataset in chunks
            for (int i = 0; i < images.Count; i += batchSize)
            {
                // Determine the actual batch size for the last mini-batch if it has fewer than batchSize items
                int currentBatchSize = Math.Min(batchSize, images.Count - i);

                // Initialize NDArray placeholders for the current mini-batch
                var batchImageArray = np.zeros(new Shape(currentBatchSize, targetWidth, targetHeight, 3), np.float32);
                var batchBboxArray = np.zeros(new Shape(currentBatchSize, 4), np.float32); // Adjust as needed for bounding boxes
                var batchLabelArray = np.zeros(new Shape(currentBatchSize, numClasses), np.float32);

                Parallel.For(0, currentBatchSize, j =>
                {
                    // Get the image and corresponding annotation
                    var image = images[i + j];
                    var annotation = annotations[i + j];

                    // Preprocess the image and bounding box
                    (NDArray processedImage, float[] scaledBbox) = PreprocessImage(
                        ImagesPath,
                        image.file_name,
                        annotation.bbox,
                        targetWidth,
                        targetHeight
                    );

                    // Assign to mini-batch arrays
                    lock (batchImageArray)
                    {
                        batchImageArray[j] = processedImage;
                    }

                    lock (batchBboxArray)
                    {
                        batchBboxArray[j] = scaledBbox;
                    }

                    lock (batchLabelArray)
                    {
                        batchLabelArray[j] = OneHotEncode(annotation.category_id, numClasses);
                    }
                });

                // Add mini-batch to the list of batches
                imageBatches.Add(batchImageArray);
                bboxBatches.Add(batchBboxArray);
                labelBatches.Add(batchLabelArray);
            }

            // Return lists of mini-batches
            return (imageBatches, bboxBatches, labelBatches);
        }
    }
}
