﻿using ICSharpCode.SharpZipLib.Zip;
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
    class ModelBuilder
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

        //define modell
        public static Functional BuildModel(int inputHeight, int inputWidth, int numClasses)
        {
            UseGPU();
            var inputs = keras.Input(shape: (inputHeight, inputWidth, 3));

            //modell definition
            var x = keras.layers.Conv2D(16, (3, 3), activation: "relu", padding: "same").Apply(inputs);
            x = keras.layers.MaxPooling2D((2, 2)).Apply(x);
            x = keras.layers.Conv2D(32, (3, 3), activation: "relu", padding: "same").Apply(x);
            x = keras.layers.MaxPooling2D((2, 2)).Apply(x);

            //ssd layer
            //var boundingBoxes = keras.layers.Conv2D(4, (3, 3), activation: "linear").Apply(x);
            var ClassScores = keras.layers.Conv2D(numClasses, (1, 1), activation: "softmax").Apply(x);


            Debug.WriteLine("\n Modell Built \n");
            return (Functional)keras.Model(inputs, ClassScores);
        }

        //Train the modell
        public static void TrainModel(Functional model, List<NDArray> trainImageBatches, List<NDArray> trainBboxBatches, List<NDArray> trainLabelBatches, int epochs, int numberOfBatch)
        {
            //var customLoss = new CustomLoss();

            // Compile model with GPU support
            UseGPU();
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
                int list_Iterator = 0;
                int batch_Iterator = 0;

                // Loop through each mini-batch while needed
                while (batch_Iterator < numberOfBatch)
                {

                    // Safeguard: Ensure `list_Iterator` is within bounds
                    if (list_Iterator >= trainImageBatches.Count)
                    {
                        // Reload new batches if we reach the end of the current batch list
                        //batch_Iterator += list_Iterator; // Increment by number of batches processed so far
                        DataLoader.LoadSavedBatchBetweenEpoch(ref trainImageBatches, ref trainLabelBatches, true, batch_Iterator);
                        list_Iterator = 0; // Reset for new batch set
                        continue; // Skip to next iteration with reloaded data
                    }

                    // Get current batch
                    var batchImages = trainImageBatches[list_Iterator];
                    var batchLabels = trainLabelBatches[list_Iterator];

                    // Train model on the current batch
                    model.fit(batchImages, batchLabels, batch_size: (int)batchImages.shape[0]);

                    Debug.WriteLine($"Batch {batch_Iterator + 1}/{numberOfBatch} completed.");
                    Console.WriteLine($"Batch {batch_Iterator + 1}/{numberOfBatch} completed.");

                    list_Iterator++; // Move to the next batch
                    batch_Iterator++; // Increment total processed batches
                }
                SaveModel(model, $"../../../../../Data/ModelBuilder/{epoch}ssd_model.keras");
                DataLoader.LoadSavedBatchBetweenEpoch(ref trainImageBatches, ref trainLabelBatches, true, 0);
                // Save the model between epoches
            }

            Debug.WriteLine("\n TRAIN COMPLETED \n");
            Console.WriteLine("\n TRAIN COMPLETED \n");

            // Save the model after training
            SaveModel(model, "../../../../../Data/ModelBuilder/ssd_model.keras");
            Debug.WriteLine("\n MODEL SAVED \n");
            Console.WriteLine("\n MODEL SAVED \n");
        }

        private static void SaveModel(Functional model, string savePath)
        {
            Directory.CreateDirectory(savePath);
            model.save(savePath);
            Debug.WriteLine("\n SAVE COMPLETED \n");
            Console.WriteLine($"ModelBuilder saved at {savePath}\n SAVE COMPLETED \n");
        }

        static public (Functional,bool) LoadModel(string modelPath,int inputHeight, int inputWidth,int numClasses)
        {
            var result = BuildModel(inputHeight, inputWidth, numClasses);
            var a = Directory.GetDirectories(modelPath, "*ssd_model.keras").Length;
            if (Directory.Exists(modelPath)&& a >0)
            {
                for (int i = 1; i <= a; i++)
                {
                    if (i==a)
                    {
                        string fullPath = Path.Combine(modelPath, $"{i}ssd_model.keras");
                        result = (Functional)keras.models.load_model(fullPath);
                        UseGPU();
                        result.compile(
                            optimizer: keras.optimizers.Adam(),
                            loss: keras.losses.CategoricalCrossentropy(),
                            metrics: new[] { "accuracy" }
                        );
                        return (result,true);
                    }
                }
            }
            return (result,false);
        }

        public static void EvaluateModel(Functional model, List<NDArray> valImageBatches, List<NDArray> valBboxBatches, List<NDArray> valLabelBatches)
        {
            float totalLoss = 0;
            int totalBatches = valImageBatches.Count;

            Debug.WriteLine("\nSTARTING EVALUATION\n");
            Console.WriteLine("\nSTARTING EVALUATION\n");
            UseGPU();
            // Iterate over each mini-batch
            for (int batch = 0; batch < totalBatches; batch++)
            {
                var batchImages = valImageBatches[batch];
                //var batchBboxes = valBboxBatches[batch];
                var batchLabels = valLabelBatches[batch];

                #region Debug
                Console.WriteLine($"Batch {batch}: Images shape: {batchImages.shape}, Labels shape: {batchLabels.shape}");


                #endregion

                // Evaluate model on the current batch
                var evaluation = model.evaluate(batchImages, batchLabels, batch_size: (int)batchImages.shape[0]);

                // Assuming the loss is stored under the key "loss" in the dictionary
                if (evaluation.ContainsKey("loss"))
                {
                    totalLoss += (float)evaluation["loss"];
                }
                
                // Dispose of the batch
                batchImages.Dispose();
                batchLabels.Dispose();
                GC.Collect();

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
            //var predictedBbox = predictions[0]; //bounding box pred
            var predictedClasses = predictions[0]; //class score pred

            //Console.WriteLine($"Predicted Bounding Box: {predictedBbox}");
            Console.WriteLine($"Predicted Class Scores: {predictedClasses}");
        }

    }
}