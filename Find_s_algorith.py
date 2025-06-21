def Find_s_algorithm(Enjoy_data):
    hypothesis=['ϕ']*(len(Enjoy_data[0])-1)

    for i in Enjoy_data:
        if i[-1].lower()=='yes':
            for j in range(len(hypothesis)):
                if hypothesis[j]=='ϕ':
                    hypothesis[j]=i[j]
                elif hypothesis[j]!=i[j]:
                    hypothesis[j]='?' 
                  
    print(hypothesis)
                
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]
print("The Most Specific Hypothesis is:")
Find_s_algorithm(data)