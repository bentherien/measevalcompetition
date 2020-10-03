filename = "C:\\Users\\Benjamin Therien\\Desktop\\testfile.txt "

println "EXECUTED";
import java.text.SimpleDateFormat;
def date = new Date();
def sdf = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
println sdf.format(date);

class CSV
{
    def headerList = [];
    def headerMap = [:];
    def csvTable = []; 

    boolean hasHead;
    CSV()
    {
        hasHead = false; 
    }
    
    CSV(String filepath, boolean header=true,String delimiter=",")
    {
        hasHead = header;
        File file = new File(filepath);
        def line, noOfLines = 0;
        file.withReader { reader ->
            while ((line = reader.readLine()) != null) {
                if(header && noOfLines == 0)
                {
                    this.headerList = line.split(delimiter);
                    int count = 0;
                    for(item in this.headerList)
                    {
                        this.headerMap[item] = count;
                        count++;
                    }
                    
                    for(String item: headerList)
                    {
                        this.csvTable.add([]);
                    }

                }
                else if (noOfLines == 0)
                {
                    hasHead = header;
                    String tempList[];
                    tempList = line.split(delimiter);
                    int count = 0;
                    for(String item: tempList)
                    {
                        this.csvTable.add([]);
                        this.csvTable[count].add(item);
                        count++;
                    }
                }
                else
                {
                    def tempList;
                    tempList = line.split(delimiter);
                    int count = 0;
                    for(String item: tempList)
                    {
                        this.csvTable[count].add(item);
                        count++;
                    }

                }
                noOfLines++
            }
    }
    }

    def get(int a, int b)
    {   if( a < 0 || b < 0 || a >= csvTable.size() ||  b >= csvTable[a].size())
            return null;
        else 
            return csvTable[a][b];
    }

    def get(String h, int b)
    {   
        int a = -1; 
        
        try {
         a = this.headerMap[h];
      } catch(Exception ex) {
        println "Invalid header name: ${h}"
        return null;
      }
        
        if( a < 0 || b < 0 || a >= csvTable.size() || b >= csvTable[a].size())
            return null;
        else 
            return csvTable[a][b];
    }

    def get(int a)
    {   if( a < 0 ||  a >= csvTable.size() )
            return null;
        else 
            return csvTable[a];
    }

    def get(String h)
    {   
        int a = -1; 
         
        try {
         a = this.headerMap[h];
      } catch(Exception ex) {
        println "Invalid header name: ${h}"
        return null;
      }

        if( a < 0 ||  a >= csvTable.size() )
            return null;
        else 
            return csvTable[a];
    }

}





eachDocument {
println doc.name+".tsv"
filepath = "C:\\Users\\Benjamin Therien\\Documents\\github\\measevalcompetition\\data\\trial\\tsv\\"
def csv = new CSV(filepath+doc.name+".tsv", true, "\t");


int count = 0;
int quantCount = 0;
annotSetName = "Measurement-${quantCount}";

for(String annotType : csv.get("annotType") )
{
    if(annotType.toString() == "Quantity")
    {
        quantCount++;
        //AnnotationSet temp = doc.getAnnotations("Measurement-${quantCount}");
        gate.Utils.addAnn(
            doc.getAnnotations("Measurement-${quantCount}"), 
            Long.valueOf((csv.get("startOffset", count) == null) ? "600" : csv.get("startOffset", count).toString()), 
            Long.valueOf((csv.get("endOffset", count) == null) ? "600" : csv.get("endOffset", count).toString()),
            "Quantity", 
            Factory.newFeatureMap());
            
          gate.Utils.addAnn(
            doc.getAnnotations("Quantity"), 
            Long.valueOf((csv.get("startOffset", count) == null) ? "600" : csv.get("startOffset", count).toString()), 
            Long.valueOf((csv.get("endOffset", count) == null) ? "600" : csv.get("endOffset", count).toString()),
            "Quantity", 
            Factory.newFeatureMap());

    }
    else if(annotType.toString() == "MeasuredEntity")
    {
        AnnotationSet temp = doc.getAnnotations("Measurement-${quantCount}");
        gate.Utils.addAnn(
            doc.getAnnotations("Measurement-${quantCount}"), 
            Long.valueOf((csv.get("startOffset", count) == null) ? "600" : csv.get("startOffset", count).toString()), 
            Long.valueOf((csv.get("endOffset", count) == null) ? "600" : csv.get("endOffset", count).toString()),
            "MeasuredEntity", 
            Factory.newFeatureMap());
            
         gate.Utils.addAnn(
            doc.getAnnotations("MeasuredEntity"), 
            Long.valueOf((csv.get("startOffset", count) == null) ? "600" : csv.get("startOffset", count).toString()), 
            Long.valueOf((csv.get("endOffset", count) == null) ? "600" : csv.get("endOffset", count).toString()),
            "MeasuredEntity", 
            Factory.newFeatureMap());
    }
    else if(annotType.toString() == "MeasuredProperty")
    {
        AnnotationSet temp = doc.getAnnotations("Measurement-${quantCount}");
        gate.Utils.addAnn(
            doc.getAnnotations("Measurement-${quantCount}"), 
            Long.valueOf((csv.get("startOffset", count) == null) ? "600" : csv.get("startOffset", count).toString()), 
            Long.valueOf((csv.get("endOffset", count) == null) ? "600" : csv.get("endOffset", count).toString()),
            "MeasuredProperty", 
            Factory.newFeatureMap());
            
         gate.Utils.addAnn(
            doc.getAnnotations("MeasuredProperty"), 
            Long.valueOf((csv.get("startOffset", count) == null) ? "600" : csv.get("startOffset", count).toString()), 
            Long.valueOf((csv.get("endOffset", count) == null) ? "600" : csv.get("endOffset", count).toString()),
            "MeasuredProperty", 
            Factory.newFeatureMap());
    }
    else if(annotType.toString() == "Qualifier")
    {
        AnnotationSet temp = doc.getAnnotations("Measurement-${quantCount}");
        gate.Utils.addAnn(
            doc.getAnnotations("Measurement-${quantCount}"), 
            Long.valueOf((csv.get("startOffset", count) == null) ? "600" : csv.get("startOffset", count).toString()), 
            Long.valueOf((csv.get("endOffset", count) == null) ? "600" : csv.get("endOffset", count).toString()),
            "Qualifier", 
            Factory.newFeatureMap());
            
        gate.Utils.addAnn(
            doc.getAnnotations("Qualifier"), 
            Long.valueOf((csv.get("startOffset", count) == null) ? "600" : csv.get("startOffset", count).toString()), 
            Long.valueOf((csv.get("endOffset", count) == null) ? "600" : csv.get("endOffset", count).toString()),
            "Qualifier", 
            Factory.newFeatureMap());
    }
    count++;
    
    
}



}

