using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DL1_Project
{
    public class CocoAnnotaiton
    {
        public class Annotation
        {
            public long id { get; set; }
            public long image_id { get; set; }
            public int category_id { get; set; }
            public List<float> bbox { get; set; }//x,y,width,height
        }

        public class Category
        {
            public int id { get; set; }
            public string name { get; set; }
        }

        public class Image
        {
            public int id { get; set; }
            public string file_name { get; set; }
            public int width { get; set; }
            public int height { get; set; }
        }
        public class Root
        {
            public List<Annotation> annotations { get; set; }
            public List<Category> categories { get; set; }
            public List<Image> images { get; set; }
        }
    }
}
