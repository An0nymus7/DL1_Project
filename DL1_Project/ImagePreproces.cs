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
        public static (NDArray, float[]) PreprocessImage(string path, List<float> bbox, int targetWidth, int targetHeight)
        {
            //kép betöltés
            Bitmap bitmap = new Bitmap(path);
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
        public static (NDArray, NDArray, NDArray) PreprocessBatch(List<CocoAnnotation.Image> images, List<CocoAnnotation.Annotation> annotations, int targetWidth, int targetHeight, int numClasses)
        { 
            var imageArray = np.zeros(new Shape(images.Count,targetWidth,targetHeight,3),np.float32);
            var bboxArray = np.zeros(new Shape(images.Count,4),np.float32);
            var labelArray = np.zeros(new Shape(images.Count,numClasses),np.float32);

            for (int i = 0; i < images.Count; i++)
            {
                var image = images[i];
                var annotation = annotations[i];

                (NDArray img, float[] scaledBbox) = PreprocessImage(image.file_name,annotation.bbox,targetWidth,targetHeight);

                imageArray[i] = img;
                bboxArray[i] = scaledBbox;
                labelArray[i] = OneHotEncode(annotation.category_id,numClasses);
            }

            return (imageArray, bboxArray, labelArray);
        }
    }
}
