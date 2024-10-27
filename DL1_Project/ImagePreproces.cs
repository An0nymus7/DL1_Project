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
    class ImagePreproces
    {
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

        public static NDArray OneHotEncode(int categoryId, int numClasses)
        {
            NDArray oneHot = np.zeros(numClasses, np.float32);
            oneHot[categoryId] = 1.0f;
            return oneHot;
        }
    }
}
