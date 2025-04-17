import pandas as pd
import os


def convert_file(folder,file_name,column_names):
    dat_file = os.path.join(folder, file_name)  
    
    if os.path.exists(dat_file):  
        try:
            df = pd.read_csv(dat_file, sep='::', engine='python', header=None, encoding='ISO-8859-1')
            
            # Rename the columns
            df.columns = column_names
            
            # Create the CSV file name by changing the extension from .dat to .csv
            csv_file = os.path.splitext(dat_file)[0] + '.csv'
            
            # Save the DataFrame to a CSV file
            df.to_csv(csv_file, index=False)
            
            print(f"Converted {dat_file} to {csv_file}")
        except Exception as e:
            print(f"Error processing {dat_file}: {e}")
    else:
        print(f"File {dat_file} does not exist.")


convert_file('./datasets','movies.dat',['ID','Movie Title','Genres'])
convert_file('./datasets','ratings.dat',['UserID',"MovieID",'Rating','TimeStamp'])
convert_file('./datasets','users.dat',['UserID','Gender','Age','Occupation','Zip-Code'])