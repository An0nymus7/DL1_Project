using System.IO;
using Newtonsoft.Json;


namespace DL1_Project
{
    class Program
    {
        static CocoAnnotation.Root LoadCocoAnnotations(string path)
        { 
            var json = File.ReadAllText(path);
            return JsonConvert.DeserializeObject<CocoAnnotation.Root>(json);
        }
        static void Main(string[] args)
        {
            var annotations = LoadCocoAnnotations("../../../../../Data/train2017/instances_train2017.json");

            Console.WriteLine($"Loaded {annotations.images.Count} images and {annotations.annotations.Count} annotations.");
        }
    }
}
