class CSV
{
    def headerList = [];
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
                    for(String item: headerList)
                    {
                        this.csvTable.add([]);
                    }

                }
                else if (noOfLines == 0)
                {
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
    {   if( a < 0 || b < 0 || a >= csvTable.size() || b >= csvTable.size())
            return null;
        else 
            return csvTable[a][b];
    }

    def get(String h, int b)
    {   
        int a = -1; 
        int count = 0;

        for(String header : headerList){
            if( header == h)
                a = count;
            count++;
        }

        // check for invalid a
        if(a == -1)
        {
            println "Invalid header name: ${h}"
            return a 
        }
        
        if( a < 0 || b < 0 || a >= csvTable.size() || b >= csvTable.size())
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
        int count = 0;

        for(String header : headerList){
            if( header == h)
                a = count;
            count++;
        }

        // check for invalid a
        if(a == -1)
        {
            println "Invalid header name: ${h}"
            return a 
        }

        if( a < 0 ||  a >= csvTable.size() )
            return null;
        else 
            return csvTable[a];
    }

}



filename = "C:\\Users\\Benjamin Therien\\Documents\\github\\measevalcompetition\\data\\trial\\tsv\\S0012821X12004384-1302.tsv"
def temp = new CSV(filename, true, "\t");
println temp.get("startOffset",5)

