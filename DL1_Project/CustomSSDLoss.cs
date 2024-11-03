using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras.Losses;

namespace DL1_Project
{
    class CustomSSDLoss : ILossFunc
    {
        private readonly ILossFunc bboxLoss;
        private readonly ILossFunc classScoreLoss;

        public CustomSSDLoss()
        {
            bboxLoss = KerasApi.keras.losses.MeanSquaredError();
            classScoreLoss = KerasApi.keras.losses.SparseCategoricalCrossentropy();
        }

        public string Reduction => throw new NotImplementedException();

        public string Name => throw new NotImplementedException();



        //custom call to calculate both losses and combines them
        public Tensor Call(Tensor yTrue, Tensor yPred) {
            var yTrueBbox = yTrue[0];
            var yTrueClass = yTrue[1];

            var yPredBbox = yPred[0];
            var yPredClass= yPred[1];

            //calculating individual loss
            var bboxLossValue = bboxLoss.Call(yTrueBbox, yPredBbox);
            var classScoreLossValue = classScoreLoss.Call(yTrueClass, yPredClass);

            //combine losses
            return bboxLossValue+classScoreLossValue;
        }

        public Tensor Call(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
        {
            throw new NotImplementedException();
        }
    }
}
