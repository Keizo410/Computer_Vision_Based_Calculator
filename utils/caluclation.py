
class Calculator:
    def __init__(self) -> None:
        self.result = ""
        self.equation = ""
    def calculate(self, expressions):
        try:
            self.set_equation(expressions)
            self.set_result()
            return self.get_result()
        except Exception as e:
            return str(e)
    
    def set_equation(self, expressions):
        for i in range(len(expressions)):
            if(expressions[i]=="_"):
                self.equation += "*"
            elif(expressions[i]=="%"):
                self.equation += "/"
            elif(expressions[i]=="]"):
                self.equation += ")"
            elif(expressions[i]=="["):
                self.equation += "("
            else:
                self.equation += expressions[i]
        
    def set_result(self):
        print("Calculating: ", self.equation)
        self.result = eval(self.equation)
    
    def get_result(self):
        return self.result
    
    def reset(self):
        self.equation = ""
        self.result = ""

            

