using ICSharpCode.SharpZipLib.Zip;
using System.Diagnostics;
using System.IO;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Losses;
using Tensorflow.NumPy;
using Tensorflow.Operations.Initializers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

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
                    {
                        tf.config.experimental.set_memory_growth(gpu, true);
                        Debug.WriteLine("\nGPU is available and memory is set!!!\n");
                        Console.WriteLine("\nGPU is available and memory is set!!!\n");
                    }
                }
                catch (Exception e)
                {
                    Debug.WriteLine($"\nFailed to initialize GPU usage: {e.Message}\n");
                    Console.WriteLine($"\nFailed to initialize GPU usage: {e.Message}\n");
                }
            }
            else
            {
                Debug.WriteLine("\nGPU is not available. Using CPU!!!!!\n");
                Console.WriteLine("\nGPU is not available. Using CPU!!!!!\n");
            }

        }

        //define SSD modell
        public static Functional BuildSSDModel(int inputHeight, int inputWidth, int numClasses)
        {
            UseGPU();
            var inputs = keras.Input(shape: (inputHeight, inputWidth, 3));

            //modell definition
            var x = keras.layers.Conv2D(32, (3, 3), activation: "relu", padding: "same").Apply(inputs);
            x = keras.layers.MaxPooling2D((2, 2)).Apply(x);
            x = keras.layers.Conv2D(64, (3, 3), activation: "relu", padding: "same").Apply(x);
            x = keras.layers.MaxPooling2D((2, 2)).Apply(x);

            //ssd layer
            //var boundingBoxes = keras.layers.Conv2D(4, (3, 3), activation: "linear").Apply(x);
            var ClassScores = keras.layers.Conv2D(numClasses, (1, 1), activation: "softmax").Apply(x);


            Debug.WriteLine("\n Modell Built \n");
            return (Functional)keras.Model(inputs,  ClassScores);
        }

        //Train the modell
        public static void TrainModel(Functional model, List<NDArray> trainImageBatches, List<NDArray> trainBboxBatches, List<NDArray> trainLabelBatches, int epochs)
        {
            var customLoss = new CustomSSDLoss();

            // Compile model with GPU support
            model.compile(
                optimizer: keras.optimizers.Adam(),
                loss: keras.losses.CategoricalCrossentropy(),
                metrics: new[] { "accuracy" }
            );

            Debug.WriteLine("\n STARTING TRAINING \n");
            Console.WriteLine("\n STARTING TRAINING \n");

            // Iterate over epochs
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                Debug.WriteLine($"\n Epoch {epoch}/{epochs} \n");
                Console.WriteLine($"\n Epoch {epoch}/{epochs} \n");

                // Loop through each mini-batch
                for (int batch = 0; batch < trainImageBatches.Count; batch++)
                {
                    var batchImages = trainImageBatches[batch];
                    //var batchBboxes = tf.convert_to_tensor(trainBboxBatches[batch]);
                    var batchLabels = trainLabelBatches[batch];


                    // Reshape labels to align with model output
                    //batchLabels = np.zeros(new Shape(batchLabels[0], 75, 75, batchLabels[1]));

                    #region asd
                    //var combinedLabels = np.concatenate(new NDArray[] { batchBboxes, batchLabels }, axis: 1);

                    //Console.WriteLine($"CombinedLabels shape: {combinedLabels.shape}");
                    //Console.WriteLine($"BatchImages shape: {batchImages.shape}");

                    //combinedLabels = (NDArray)tf.convert_to_tensor(combinedLabels); 
                    #endregion

                    Console.WriteLine($"BatchImages shape: {batchImages.shape}");
                    Console.WriteLine($"BatchLabels shape: {batchLabels.shape}");
                    Console.WriteLine($"Model input shape: {model.inputs.shape}");
                    //Console.WriteLine($"Model output shape: {model.OutputShape.GetShape()}");


                    // Fit model on the current batch
                    model.fit(batchImages, batchLabels, batch_size: (int)batchImages.shape[0]);

                    Debug.WriteLine($"Batch {batch + 1}/{trainImageBatches.Count} completed.");
                    Console.WriteLine($"Batch {batch + 1}/{trainImageBatches.Count} completed.");
                }
            }

            Debug.WriteLine("\n TRAIN COMPLETED \n");
            Console.WriteLine("\n TRAIN COMPLETED \n");

            // Save the model after training
            SaveModel(model, "../../../../../Data/Model/ssd_model");
            Debug.WriteLine("\n MODEL SAVED \n");
            Console.WriteLine("\n MODEL SAVED \n");
        }

        private static void SaveModel(Functional model, string savePath)
        {
            Directory.CreateDirectory(savePath);
            model.save(savePath);

            Debug.WriteLine("\n SAVE COMPLETED \n");
            Console.WriteLine($"Model saved at {savePath}\n SAVE COMPLETED \n");
        }

        public static void EvaluateModel(Functional model, List<NDArray> valImageBatches, List<NDArray> valBboxBatches, List<NDArray> valLabelBatches)
        {
            float totalLoss = 0;
            int totalBatches = valImageBatches.Count;

            Debug.WriteLine("\nSTARTING EVALUATION\n");
            Console.WriteLine("\nSTARTING EVALUATION\n");

            // Iterate over each mini-batch
            for (int batch = 0; batch < totalBatches; batch++)
            {
                var batchImages = valImageBatches[batch];
                var batchBboxes = valBboxBatches[batch];
                var batchLabels = valLabelBatches[batch];

                // Evaluate model on the current batch
                var evaluation = model.evaluate(batchImages, (NDArray)new Tensors(batchBboxes, batchLabels));

                // Assuming the loss is stored under the key "loss" in the dictionary
                if (evaluation.ContainsKey("loss"))
                {
                    totalLoss += (float)evaluation["loss"];
                }

                Debug.WriteLine($"Batch {batch + 1}/{totalBatches} evaluated.");
                Console.WriteLine($"Batch {batch + 1}/{totalBatches} evaluated.");
            }

            // Calculate the average loss over all batches
            float averageLoss = totalLoss / totalBatches;

            Debug.WriteLine($"\nEvaluation Complete. Average Loss: {averageLoss}\n");
            Console.WriteLine($"\nEvaluation Complete. Average Loss: {averageLoss}\n");
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
