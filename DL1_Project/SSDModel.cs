using ICSharpCode.SharpZipLib.Zip;
using System.Diagnostics;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Losses;
using Tensorflow.NumPy;
using Tensorflow.Operations.Initializers;
using static Tensorflow.Binding;

namespace DL1_Project
{
    class SSDModel
    {
        //use GPU if available
        public static void UseGPU()
        {
            var gpus = tf.config.list_physical_devices("GPU");
            if (gpus.Length > 0)
            {
                try
                {
                    //set memory growth on GPU
                    foreach (var gpu in gpus)
                        tf.config.experimental.set_memory_growth(gpu, true);
                    Debug.WriteLine("\nGPU is available and memory is set!!!\n");
                }
                catch (Exception e)
                {
                    Debug.WriteLine($"\nFailed to initialize GPU usage: {e.Message}\n");
                }
            }
            else
            {
                Debug.WriteLine("\nGPU is not available. Using CPU!!!!!\n");
            }

        }

        //define SSD modell
        public static Functional BuildSSDModel(int inputHeight, int inputWidth, int numClasses)
        {
            UseGPU();
            var inputs = KerasApi.keras.Input(shape: (inputHeight, inputWidth, 3));

            //modell definition
            var x = KerasApi.keras.layers.Conv2D(32, (3, 3), activation: "relu", padding: "same").Apply(inputs);
            x = KerasApi.keras.layers.MaxPooling2D((2, 2)).Apply(x);
            x = KerasApi.keras.layers.Conv2D(64, (3, 3), activation: "relu", padding: "same").Apply(x);
            x = KerasApi.keras.layers.MaxPooling2D((2, 2)).Apply(x);

            //ssd layer
            var boundingBoxes = KerasApi.keras.layers.Conv2D(4, (3, 3), activation: "linear").Apply(x);
            var ClassScores = KerasApi.keras.layers.Conv2D(numClasses, (3, 3), activation: "softmax").Apply(x);

            return (Functional)KerasApi.keras.Model(inputs, new Tensors(boundingBoxes, ClassScores));
        }

        //Train the modell
        public static void TrainModel(Functional model, NDArray trainImages, NDArray trainBboxes, NDArray trainLabels)
        {
            var CustomLoss = new CustomSSDLoss();

            //compile modell with GPU support
            model.compile(
            optimizer: KerasApi.keras.optimizers.Adam(),
            loss: CustomLoss);
            //work around because it needs only one Iloss thingy so we define a custom loss function to calculate both

            //train
            model.fit(trainImages, (NDArray)new Tensors(trainBboxes, trainLabels), epochs: 10, batch_size: 32);

            Debug.WriteLine("\n TRAIN COMPLETED \n");
            Console.WriteLine("\n TRAIN COMPLETED \n");

            //save the model
            Directory.CreateDirectory("../../../../../Data");
            model.save("../../../../../Data");
            Debug.WriteLine("\n SAVE COMPLETED \n");
            Console.WriteLine("\n SAVE COMPLETED \n");

        }

        public static void EvaluateModel(Functional model, NDArray valImages, NDArray valBboxes, NDArray valLabels)
        {
            var evaluation = model.evaluate(valImages, new Tensors(valBboxes, valLabels));
            Debug.WriteLine($"\nEvaluation Loss: {evaluation}\n");
            Console.WriteLine($"\nEvaluation Loss: {evaluation}\n");
        }

        public static void Predict(Functional model, NDArray image)
        {
            var predictions = model.predict(image);
            var predictedBbox = predictions[0]; //bounding box pred
            var predictedClasses = predictions[1]; //class score pred

            Console.WriteLine($"Predicted Bounding Box: {predictedBbox}");
            Console.WriteLine($"Predicted Class Scores: {predictedClasses}");
        }

    }
}
