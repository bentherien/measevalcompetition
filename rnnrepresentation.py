class RNNRepresentation(nn.Module):


    def __init__(self,dep_to_ix,tag_to_ix,emb_dim=50,d_model=50):
        """
        tag_to_ix: dictionary mapping all possible output classes to numbers
        dep_to_ix: dictionary mapping all possible dependencies to numbers
        d_model: dimension of the hidden layer of the LSTM and, therefore, its output size
        emb_dim: dimension of the dependency embeddings you wish to create 
        """
        super().__init__()
        self.tag_to_ix = tag_to_ix
        self.dep_to_ix = dep_to_ix
        self.d_model = d_model


        self.depEmbedding = nn.Embedding(len(dep_to_ix), emb_dim)
        self.lstm = nn.LSTM(emb_dim,d_model,dropout=0,bidirectional=False)
        self.out = nn.Linear(d_model,len(tag_to_ix))


    def forward(self, input, CUDA):
        """
        input: a path of dependecies you wish to represent
        """
        def convert_2_tensor(seq, to_ix, dt, CUDA):
            if to_ix == None:
                return Var(torch.tensor(seq, dtype=dt))
            else:
                idxs = list()
                for w in seq:
                    if w in to_ix:
                        idxs.append(to_ix[w])
                return Var(torch.tensor(idxs, dtype=dt),CUDA)

        def getLstmLast(ip):
            lstm_out = self.lstm(ip)
            if lstm_out[0].shape[1] == 1:
                lstm_finalLayer = lstm_out[0].squeeze()
                lstm_finalLayer = lstm_finalLayer.unsqueeze(0)
            else:
                lstm_finalLayer = lstm_out[0].squeeze()[-1:,]

            return lstm_finalLayer

        
        pathRep = self.depEmbedding(convert_2_tensor(depPath,self.dep_to_ix,torch.long,CUDA)).unsqueeze(0) 
        lstmLast = getLstmLast(pathRep)
        out = self.out(lstmLast)
        pred = F.softmax(out, dim=1)

        
        return pred, out
    

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        temp = torch.load(filepath)
        self.tag_to_ix = temp["tag_to_ix"]
        self.dep_to_ix = temp["dep_to_ix"]
        self.load_state_dict(temp["state_dict"])

    def save(self,filepath):
        """
        save model to a binary file
        """
        temp = {
            "dep_to_ix":self.dep_to_ix,
            "tag_to_ix":self.tag_to_ix,
            "state_dict": self.state_dict()
        }
        torch.save(temp,filepath)


    def predict(self,input,CUDA):
        """
        Preditions of the model on testdata
        """
        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():
            all_predictions = []




            # you must rewite the code below



            _,prediction = self(input,CUDA)
            _,maxIndices = torch.max(prediction,2)
            for x in [z.cpu().detach().numpy() for z in maxIndices]:
                all_predictions.append([inv_dic[y] for y in x])

            
        return all_predictions

