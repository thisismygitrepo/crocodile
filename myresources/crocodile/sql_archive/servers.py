"""
servers' details that Alex has access to so far.


Thursday, 30 December 2021
2:49 PM

"""
"""
Quetions: 
What is exactly is the difference between dobipub from CDW and prod from CMP
"""


class BaseServer:
    username = r"HAD_ID" 
    password = r"HAD_ID"

    @property
    def odbc_creds(self):
        if self.username == "HAD_ID":
            return "Trusted_Connection=Yes"  # uses windows authentication, which is same as HAD login.
        else:
            return f"UID={self.username};PWD={self.password}"


class Clinipi(BaseServer):
    """
    Local server for the team: mainly used for dashboard purposes.
    Local SQL instance
    """
    servername = r"Sah0004344\salhn_clinepi"  # sah0004344/SALHN_ClinEpi
    serverid = r"10.18.203.81\salhn_clinepi"
    username = r"SALHN_DASHBOARD_APP"
    password = r"D@shboard2019!"


class Empi(BaseServer):
    """
    A server to map IDs to other systems
    """
    servername = r"hltcls001empi"
    username = r"SALHN_RO"
    password = r"Gjs#11knaKqpONe"


class CDW(BaseServer):
    """
    Flinders data going back about 4 (5?) years
    """
    servername = r"HLT142SQL002\CDW_P_2012"
    username = r"SALHN_Tableau"
    password = r"$@LtAB1!"
    dbs = [r"dobipub01"]


class EMR_FMC(BaseServer):
    """Flinders (FMC) has only been using EMR since 3/3/21 so has only 9 months of EMR data.
    OCMIO CDAP database
    """
    servername = r"cmpproddb001"
    dbs = [r"PROD",  #  copy of EMR
          ]
    
 

 
