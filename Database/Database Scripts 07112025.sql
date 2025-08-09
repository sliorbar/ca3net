USE [CUNY]
GO
/****** Object:  Table [dbo].[spike_times_Archive]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[spike_times_Archive](
	[index] [bigint] NULL,
	[0] [float] NULL,
	[SelectedPC] [varchar](255) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[spike_times]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[spike_times](
	[index] [bigint] NULL,
	[0] [float] NULL,
	[SelectedPC] [varchar](255) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Experiments_Archive]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Experiments_Archive](
	[ID] [int] NOT NULL,
	[Description] [nvarchar](250) NULL,
	[Param JSON] [nvarchar](max) NULL,
	[Start Timestamp] [datetime] NULL,
	[Finish Timestamp] [datetime] NULL,
	[DeArchive] [bit] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  View [dbo].[vwSpikeTimes]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


create VIEW [dbo].[vwSpikeTimes]
AS
SELECT        [index], [0], SelectedPC, expid
FROM            spike_times
union all
SELECT        a.[index], a.[0], a.SelectedPC, a.expid
FROM            spike_times_Archive AS a INNER JOIN
                         Experiments_Archive AS b ON a.expid = b.ID
WHERE        (b.DeArchive = 1)
GO
/****** Object:  Table [dbo].[spiking_neurons]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[spiking_neurons](
	[index] [bigint] NULL,
	[0] [int] NULL,
	[SelectedPC] [varchar](255) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[spiking_neurons_Archive]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[spiking_neurons_Archive](
	[index] [bigint] NULL,
	[0] [int] NULL,
	[SelectedPC] [varchar](max) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  View [dbo].[vwSpikingNeurons]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


create VIEW [dbo].[vwSpikingNeurons]
AS
SELECT        [index], [0], SelectedPC, expid
FROM            spiking_neurons
union all
SELECT        a.[index], a.[0], a.SelectedPC, a.expid
FROM            spiking_neurons_Archive AS a INNER JOIN
                         Experiments_Archive AS b ON a.expid = b.ID
WHERE        (b.DeArchive = 1)
GO
/****** Object:  Table [dbo].[wexc_s_Archive]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[wexc_s_Archive](
	[index] [float] NULL,
	[InputFromPC] [bigint] NULL,
	[value] [float] NULL,
	[SelectedPC] [bigint] NULL,
	[offset] [bigint] NULL,
	[expid] [bigint] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[wexc_s]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[wexc_s](
	[index] [float] NULL,
	[InputFromPC] [bigint] NULL,
	[value] [float] NULL,
	[SelectedPC] [bigint] NULL,
	[offset] [bigint] NULL,
	[expid] [bigint] NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[vwSpikingPCFilter]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE VIEW [dbo].[vwSpikingPCFilter]
AS
SELECT DISTINCT InputFromPC, expid
FROM            dbo.wexc_s
Union all 

SELECT      DISTINCT  a.InputFromPC, a.expid
FROM            wexc_s_Archive AS a INNER JOIN
                         Experiments_Archive AS b ON a.expid = b.ID
WHERE        (b.DeArchive = 1)
GO
/****** Object:  View [dbo].[vwSpike_PC_Times]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE VIEW [dbo].[vwSpike_PC_Times]
AS
SELECT        a.[index], b.[0] AS [PC Num], a.[0] AS Spike_Time, CAST(a.[0] AS int) AS Spike_Time_int, a.expid
FROM            vwSpikeTimes AS a INNER JOIN
                         vwSpikingNeurons AS b ON a.[index] = b.[index] AND a.expid = b.expid INNER JOIN
                         vwSpikingPCFilter AS c ON b.expid = c.expid AND b.[0] = c.InputFromPC
GO
/****** Object:  Table [dbo].[BCs]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[BCs](
	[t] [float] NOT NULL,
	[BC] [int] NOT NULL,
	[expid] [int] NOT NULL,
	[Previous_Spike_Time] [float] NULL,
PRIMARY KEY CLUSTERED 
(
	[expid] ASC,
	[BC] ASC,
	[t] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[BCs_Archive]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[BCs_Archive](
	[t] [float] NULL,
	[BC] [int] NULL,
	[expid] [int] NOT NULL,
	[Previous_Spike_Time] [float] NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[vwSpike_PC_Times - All BCs]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO








CREATE VIEW [dbo].[vwSpike_PC_Times - All BCs]
AS
SELECT        t, BC, expid, Previous_Spike_Time
FROM            BCs
UNION ALL
SELECT        t, BC, expid, Previous_Spike_Time
FROM            BCs_Archive
GO
/****** Object:  Table [dbo].[Experiments Parameters]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Experiments Parameters](
	[ExpID] [int] NULL,
	[adaptation_mult] [varchar](max) NULL,
	[am] [varchar](max) NULL,
	[ap] [varchar](max) NULL,
	[connection_prob_bc] [varchar](max) NULL,
	[connection_prob_pc] [varchar](max) NULL,
	[cue] [varchar](max) NULL,
	[cue_start] [varchar](max) NULL,
	[learning_rate] [varchar](max) NULL,
	[selected_pc] [varchar](max) NULL,
	[stdp_mode] [varchar](max) NULL,
	[synaptic_delay] [varchar](max) NULL,
	[taum] [varchar](max) NULL,
	[taup] [varchar](max) NULL,
	[total_duration] [varchar](max) NULL,
	[post_AuC] [varchar](max) NULL,
	[pre_AuC] [varchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[replays]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[replays](
	[ExpID] [int] NULL,
	[Start] [float] NULL,
	[End] [float] NULL,
	[Slope] [float] NULL,
	[min_range] [float] NULL,
	[max_range] [float] NULL,
	[Duration] [float] NULL,
	[R_Value] [float] NULL,
	[P_Value] [float] NULL,
	[Std_Err] [float] NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[vwReplaysWithIndices]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [dbo].[vwReplaysWithIndices] AS
WITH NumberedReplays AS (
    SELECT 
        ExpID,
        [Start],
        [End],
        Slope,
        min_range,
        max_range,
        Duration,
        R_Value,
        P_Value,
        Std_Err,
        ROW_NUMBER() OVER (PARTITION BY ExpID ORDER BY [Start]) AS CurrentIndex
    FROM 
        Replays
)
SELECT 
    ExpID,
    [Start],
    [End],
    Slope,
    min_range,
    max_range,
    Duration,
    R_Value,
    P_Value,
    Std_Err,
    CurrentIndex,
    LAG(CurrentIndex, 1) OVER (PARTITION BY ExpID ORDER BY CurrentIndex) AS PreviousIndex
FROM 
    NumberedReplays;



GO
/****** Object:  View [dbo].[vwReplayAnalysis]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [dbo].[vwReplayAnalysis]
AS
SELECT 
    replays.ExpID, 
    replays.[Start], 
    replays.[End], 
    replays.Slope, 
    replays.min_range, 
    replays.max_range, 
    replays.Duration, 
    replays.R_Value, 
    replays.P_Value, 
    replays.Std_Err, 
    replays.CurrentIndex, 
    replays.PreviousIndex, 
    p_replays.Duration AS p_Duration, 
    p_replays.Slope AS p_Slope,
    CASE 
        WHEN p_replays.Slope * replays.Slope > 0 THEN 1 
        ELSE -1 
    END AS Seq_Reply_Direction
	FROM            dbo.vwReplaysWithIndices AS replays LEFT OUTER JOIN
                         dbo.vwReplaysWithIndices AS p_replays ON replays.PreviousIndex = p_replays.CurrentIndex AND replays.ExpID = p_replays.ExpID
GO
/****** Object:  View [dbo].[vwReplaySpeedAnalysis]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [dbo].[vwReplaySpeedAnalysis]
AS
SELECT   rep.ExpID, rep.Start, rep.[End], rep.Slope, rep.min_range, rep.max_range, rep.Duration, rep.[End] - rep.Start AS [replay duration total], [Experiments Parameters].am, 
                         [Experiments Parameters].ap, CASE WHEN (rep.Slope > 0) THEN 'F' ELSE 'R' END AS Direction, rep.p_Duration, rep.p_Slope, rep.Seq_Reply_Direction
FROM         vwReplayAnalysis AS rep INNER JOIN
                         [Experiments Parameters] ON rep.ExpID = [Experiments Parameters].ExpID
--WHERE     (rep.[End] - rep.Start > 400)
GO
/****** Object:  View [dbo].[vwSpike_PC_Times - All PCs - old]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO






CREATE VIEW [dbo].[vwSpike_PC_Times - All PCs - old]
AS
SELECT        a.[index], b.[0] AS [PC Num], a.[0] AS Spike_Time, CAST(a.[0] AS int) AS Spike_Time_int, a.expid, c.InputFromPC AS [PC Filter]
FROM            vwSpikeTimes AS a INNER JOIN
                         vwSpikingNeurons AS b ON a.[index] = b.[index] AND a.expid = b.expid LEFT OUTER JOIN
                         vwSpikingPCFilter AS c ON b.expid = c.expid AND b.[0] = c.InputFromPC					 
--where dbo.spike_times.expid in (select top (12) id from Experiments order by id desc)
						  
GO
/****** Object:  Table [dbo].[Experiments]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Experiments](
	[ID] [int] IDENTITY(1,1) NOT NULL,
	[Description] [nvarchar](250) NULL,
	[Param JSON] [nvarchar](max) NULL,
	[Start Timestamp] [datetime] NULL,
	[Finish Timestamp] [datetime] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  View [dbo].[vwExperiments]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [dbo].[vwExperiments]
AS
SELECT        ID, Description, [Param JSON], [Start Timestamp], [Finish Timestamp]
FROM            Experiments
union all
SELECT        ID, Description, [Param JSON], [Start Timestamp], [Finish Timestamp]
FROM            Experiments_Archive AS b
WHERE        (b.DeArchive = 1)
GO
/****** Object:  View [dbo].[vwSpike_PC_Times - Last Spikes]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO






Create VIEW [dbo].[vwSpike_PC_Times - Last Spikes]
AS
/*
SELECT        a.[index], b.[0] AS [PC Num], a.[0] AS Spike_Time, CAST(a.[0] AS int) AS Spike_Time_int, a.expid, c.InputFromPC AS [PC Filter]
FROM            vwSpikeTimes AS a INNER JOIN
                         vwSpikingNeurons AS b ON a.[index] = b.[index] AND a.expid = b.expid LEFT OUTER JOIN
                         vwSpikingPCFilter AS c ON b.expid = c.expid AND b.[0] = c.InputFromPC			
	
*/
--where dbo.spike_times.expid in (select top (12) id from Experiments order by id desc)
/*
WITH SpikeData AS (
    SELECT 
        a.[index], 
        b.[0] AS [PC Num], 
        a.[0] AS Spike_Time, 
        CAST(a.[0] AS int) AS Spike_Time_int, 
        a.expid, 
        c.InputFromPC AS [PC Filter],
        -- Use LAG to get the previous spike time within the same experiment and PC
        LAG(CAST(a.[0] AS int)) OVER (PARTITION BY a.expid, b.[0] ORDER BY CAST(a.[0] AS int)) AS Last_Spike_Time_int
    FROM vwSpikeTimes AS a
    INNER JOIN vwSpikingNeurons AS b ON a.[index] = b.[index] AND a.expid = b.expid
    LEFT OUTER JOIN vwSpikingPCFilter AS c ON b.expid = c.expid AND b.[0] = c.InputFromPC
)
SELECT * FROM SpikeData;
*/						
SELECT 
    ROW_NUMBER() OVER (ORDER BY a.expid, b.[0], CAST(a.[0] AS int)) AS RowNum,  -- Row Number sorted by ExpID, PC Num, and Spike Time Int
    a.[index], 
    b.[0] AS [PC Num], 
    a.[0] AS Spike_Time, 
    CAST(a.[0] AS int) AS Spike_Time_int, 
    a.expid, 
    c.InputFromPC AS [PC Filter]
FROM 
    vwSpikeTimes AS a 
INNER JOIN 
    vwSpikingNeurons AS b ON a.[index] = b.[index] AND a.expid = b.expid 
LEFT OUTER JOIN 
    vwSpikingPCFilter AS c ON b.expid = c.expid AND b.[0] = c.InputFromPC;
GO
/****** Object:  View [dbo].[vwFailedExp]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [dbo].[vwFailedExp]
AS
SELECT        ID, Description, [Param JSON], [Start Timestamp], [Finish Timestamp]
FROM            Experiments
WHERE        ([Finish Timestamp] IS NULL)
GO
/****** Object:  Table [dbo].[SynWeightsStats_Archive]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[SynWeightsStats_Archive](
	[Bucket] [float] NULL,
	[variable] [bigint] NULL,
	[value] [bigint] NULL,
	[SelectedPC] [bigint] NULL,
	[expid] [bigint] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[SynWeightsStats]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[SynWeightsStats](
	[Bucket] [float] NULL,
	[variable] [bigint] NULL,
	[value] [bigint] NULL,
	[SelectedPC] [bigint] NULL,
	[expid] [bigint] NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[vwSynWeightsStats]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

create VIEW [dbo].[vwSynWeightsStats]
AS
SELECT        *
FROM            SynWeightsStats
union all
SELECT        a.*
FROM            SynWeightsStats_Archive AS a INNER JOIN
                         Experiments_Archive AS b ON a.expid = b.ID
WHERE        (b.DeArchive = 1)
GO
/****** Object:  View [dbo].[vwExperimentsAll]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE VIEW [dbo].[vwExperimentsAll]
AS
SELECT        ID, Description, [Param JSON], [Start Timestamp], [Finish Timestamp], 1 as DeArchive
FROM            Experiments
union all
SELECT        ID, Description, [Param JSON], [Start Timestamp], [Finish Timestamp], DeArchive
FROM            Experiments_Archive AS b
--WHERE        (b.DeArchive = 1)
GO
/****** Object:  View [dbo].[vwSynapticChanges]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [dbo].[vwSynapticChanges]
AS
SELECT        [index], InputFromPC, value, SelectedPC, offset, expid, offset + [index] AS time_ms
FROM            dbo.wexc_s

union all
SELECT        a.[index], a.InputFromPC, a.value, a.SelectedPC, a.offset, a.expid, a.offset + a.[index] AS time_ms
FROM            wexc_s_Archive AS a INNER JOIN
                         Experiments_Archive AS b ON a.expid = b.ID
WHERE        (b.DeArchive = 1)
GO
/****** Object:  View [dbo].[vwSpikesByExp]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO



/****** Script for SelectTopNRows command from SSMS  ******/
CREATE view [dbo].[vwSpikesByExp] as 
SELECT        a.expid, b.Description, COUNT(a.[index]) AS [Num of spikes]
FROM            vwspiketimes AS a INNER JOIN
                         vwExperiments AS b ON a.expid = b.ID
GROUP BY a.expid, b.Description
--ORDER BY expid DESC
GO
/****** Object:  Table [dbo].[SpikeData]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[SpikeData](
	[ExpID] [int] NOT NULL,
	[PC_ID] [int] NOT NULL,
	[Spike_Time] [float] NOT NULL,
	[Previous_Spike_Time] [float] NULL,
PRIMARY KEY CLUSTERED 
(
	[ExpID] ASC,
	[PC_ID] ASC,
	[Spike_Time] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[SpikeData_Archive]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[SpikeData_Archive](
	[ExpID] [int] NOT NULL,
	[PC_ID] [int] NOT NULL,
	[Spike_Time] [float] NOT NULL,
	[Previous_Spike_Time] [float] NULL,
PRIMARY KEY CLUSTERED 
(
	[ExpID] ASC,
	[PC_ID] ASC,
	[Spike_Time] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[vwSpike_PC_Times - All PCs]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO










CREATE VIEW [dbo].[vwSpike_PC_Times - All PCs]
AS

SELECT        ExpID, PC_ID, Spike_Time, Previous_Spike_Time
FROM            SpikeData
union all
SELECT        ExpID, PC_ID, Spike_Time, Previous_Spike_Time
FROM            SpikeData_Archive
	

			
GO
/****** Object:  Table [dbo].[PSCs]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PSCs](
	[index] [bigint] NULL,
	[t] [float] NULL,
	[PC] [int] NULL,
	[SelectedPC] [varchar](max) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PSCs_Archive]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PSCs_Archive](
	[index] [bigint] NULL,
	[t] [float] NULL,
	[PC] [int] NULL,
	[SelectedPC] [varchar](max) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[rate]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[rate](
	[index] [bigint] NULL,
	[0] [float] NULL,
	[SelectedPC] [varchar](max) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[rate_Archive]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[rate_Archive](
	[index] [bigint] NULL,
	[0] [float] NULL,
	[SelectedPC] [varchar](max) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[replays_end_section]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[replays_end_section](
	[ExpID] [bigint] NULL,
	[Start] [float] NULL,
	[End] [float] NULL,
	[Slope] [float] NULL,
	[BCs cnt] [bigint] NULL,
	[MinPC] [bigint] NULL,
	[MaxPC] [bigint] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Syn_weights_changes]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Syn_weights_changes](
	[row_bin] [varchar](max) NULL,
	[col_bin] [varchar](max) NULL,
	[start_bin] [varchar](max) NULL,
	[end_bin] [varchar](max) NULL,
	[counts] [bigint] NULL,
	[ExpID] [varchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Syn_weights_changes_averages]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Syn_weights_changes_averages](
	[ExpID] [varchar](max) NULL,
	[Mean] [float] NULL,
	[Mean_change_per.] [float] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Syn_weights_Hist]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Syn_weights_Hist](
	[Folder_Identifier] [bigint] NULL,
	[Bin_Description] [varchar](max) NULL,
	[Bin_0] [float] NULL,
	[Bin_1] [float] NULL,
	[Bin_2] [float] NULL,
	[Bin_3] [float] NULL,
	[Bin_4] [float] NULL,
	[Bin_5] [float] NULL,
	[Bin_6] [float] NULL,
	[Bin_7] [float] NULL,
	[Bin_8] [float] NULL,
	[Bin_9] [float] NULL,
	[Bin_10] [float] NULL,
	[Bin_11] [float] NULL,
	[Bin_12] [float] NULL,
	[Bin_13] [float] NULL,
	[Bin_14] [float] NULL,
	[Bin_15] [float] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  StoredProcedure [dbo].[ArchiveExp]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO







-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
CREATE PROCEDURE [dbo].[ArchiveExp]
 @ExpID as int
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
insert into [dbo].[SpikeData_Archive] select * from [dbo].[SpikeData] where expid = (@ExpID)
--insert into [dbo].[PSCs_archive] select * from [dbo].[PSCs] where expid = (@ExpID)
insert into [dbo].[SynWeightsStats_archive] select * from [dbo].[SynWeightsStats] where expid = (@ExpID)
insert into [dbo].[BCs_archive] select * from [dbo].[BCs] where expid = (@ExpID)
insert into [dbo].[rate_archive] select * from [dbo].[rate] where expid = (@ExpID)
--insert into [dbo].[spike_times_archive] select * from [dbo].[spike_times] where expid = (@ExpID)
--insert into [dbo].[spiking_neurons_archive] select * from [dbo].[spiking_neurons] where expid = (@ExpID)
insert into [dbo].[wexc_s_archive] select * from [dbo].[wexc_s] where expid = (@ExpID)
insert into [dbo].[Experiments_archive] select *, null from [dbo].[Experiments] where id = (@ExpID)
exec [dbo].[CleanUpExp] @ExpID
Update [dbo].[Experiments_archive] set [DeArchive] = 1 where [ID] = @ExpID 
END
GO
/****** Object:  StoredProcedure [dbo].[ArchiveExpRange]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE PROCEDURE [dbo].[ArchiveExpRange]
    @StartExpID INT,
    @EndExpID INT
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @ExpID INT = @StartExpID;

    WHILE @ExpID <= @EndExpID
    BEGIN
        BEGIN TRY
            -- Call your existing archive procedure
            EXEC dbo.ArchiveExp @ExpID;
        END TRY
        BEGIN CATCH
            PRINT 'Error archiving ExpID = ' + CAST(@ExpID AS VARCHAR) +
                  '. Error: ' + ERROR_MESSAGE();
            -- You could optionally log the error into an error log table here
        END CATCH

        SET @ExpID = @ExpID + 1;
    END
END;
GO
/****** Object:  StoredProcedure [dbo].[CleanUp]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
CREATE PROCEDURE [dbo].[CleanUp]

AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
delete from [dbo].[PSCs]
where expid in (select [id] from [dbo].[vwFailedExp])
delete from [dbo].[SynWeightsStats]
where expid in (select [id] from [dbo].[vwFailedExp])
delete from [dbo].[BCs]
where expid in (select [id] from [dbo].[vwFailedExp])
delete from [dbo].[rate]
where expid in (select [id] from [dbo].[vwFailedExp])
delete from [dbo].[SpikeData]
where expid in (select [id] from [dbo].[vwFailedExp])
delete from [dbo].[wexc_s]
where expid in (select [id] from [dbo].[vwFailedExp])
delete from [dbo].[Experiments]
where id in (select [id] from [dbo].[vwFailedExp])
END
GO
/****** Object:  StoredProcedure [dbo].[CleanUpExp]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO



-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
CREATE PROCEDURE [dbo].[CleanUpExp]
 @ExpID as int
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
delete from [dbo].[SynWeightsStats]
where expid = (@ExpID)
delete from [dbo].[BCs]
where expid = (@ExpID)
delete from [dbo].[rate]
where expid = (@ExpID)
delete from [dbo].[SpikeData]
where expid = (@ExpID)
delete from [dbo].[wexc_s]
where expid = (@ExpID)
delete from [dbo].[Experiments]
where id = (@ExpID)
END
GO
/****** Object:  StoredProcedure [dbo].[DeArchiveExp]    Script Date: 7/11/2025 9:58:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO





-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
CREATE PROCEDURE [dbo].[DeArchiveExp]
 @ExpID as int
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
Update [dbo].[Experiments_Archive] set [DeArchive] = 1 where [ID] = @ExpID
END
GO
EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPane1', @value=N'[0E232FF0-B466-11cf-A24F-00AA00A3EFFF, 1.00]
Begin DesignProperties = 
   Begin PaneConfigurations = 
      Begin PaneConfiguration = 0
         NumPanes = 4
         Configuration = "(H (1[40] 4[20] 2[20] 3) )"
      End
      Begin PaneConfiguration = 1
         NumPanes = 3
         Configuration = "(H (1 [50] 4 [25] 3))"
      End
      Begin PaneConfiguration = 2
         NumPanes = 3
         Configuration = "(H (1 [50] 2 [25] 3))"
      End
      Begin PaneConfiguration = 3
         NumPanes = 3
         Configuration = "(H (4 [30] 2 [40] 3))"
      End
      Begin PaneConfiguration = 4
         NumPanes = 2
         Configuration = "(H (1 [56] 3))"
      End
      Begin PaneConfiguration = 5
         NumPanes = 2
         Configuration = "(H (2 [66] 3))"
      End
      Begin PaneConfiguration = 6
         NumPanes = 2
         Configuration = "(H (4 [50] 3))"
      End
      Begin PaneConfiguration = 7
         NumPanes = 1
         Configuration = "(V (3))"
      End
      Begin PaneConfiguration = 8
         NumPanes = 3
         Configuration = "(H (1[56] 4[18] 2) )"
      End
      Begin PaneConfiguration = 9
         NumPanes = 2
         Configuration = "(H (1 [75] 4))"
      End
      Begin PaneConfiguration = 10
         NumPanes = 2
         Configuration = "(H (1[66] 2) )"
      End
      Begin PaneConfiguration = 11
         NumPanes = 2
         Configuration = "(H (4 [60] 2))"
      End
      Begin PaneConfiguration = 12
         NumPanes = 1
         Configuration = "(H (1) )"
      End
      Begin PaneConfiguration = 13
         NumPanes = 1
         Configuration = "(V (4))"
      End
      Begin PaneConfiguration = 14
         NumPanes = 1
         Configuration = "(V (2))"
      End
      ActivePaneConfig = 0
   End
   Begin DiagramPane = 
      Begin Origin = 
         Top = 0
         Left = 0
      End
      Begin Tables = 
         Begin Table = "Experiments"
            Begin Extent = 
               Top = 6
               Left = 38
               Bottom = 136
               Right = 220
            End
            DisplayFlags = 280
            TopColumn = 0
         End
      End
   End
   Begin SQLPane = 
   End
   Begin DataPane = 
      Begin ParameterDefaults = ""
      End
   End
   Begin CriteriaPane = 
      Begin ColumnWidths = 11
         Column = 1440
         Alias = 900
         Table = 1170
         Output = 720
         Append = 1400
         NewValue = 1170
         SortType = 1350
         SortOrder = 1410
         GroupBy = 1350
         Filter = 1350
         Or = 1350
         Or = 1350
         Or = 1350
      End
   End
End
' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'vwExperiments'
GO
EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPaneCount', @value=1 , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'vwExperiments'
GO
EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPane1', @value=N'[0E232FF0-B466-11cf-A24F-00AA00A3EFFF, 1.00]
Begin DesignProperties = 
   Begin PaneConfigurations = 
      Begin PaneConfiguration = 0
         NumPanes = 4
         Configuration = "(H (1[40] 4[20] 2[20] 3) )"
      End
      Begin PaneConfiguration = 1
         NumPanes = 3
         Configuration = "(H (1 [50] 4 [25] 3))"
      End
      Begin PaneConfiguration = 2
         NumPanes = 3
         Configuration = "(H (1 [50] 2 [25] 3))"
      End
      Begin PaneConfiguration = 3
         NumPanes = 3
         Configuration = "(H (4 [30] 2 [40] 3))"
      End
      Begin PaneConfiguration = 4
         NumPanes = 2
         Configuration = "(H (1 [56] 3))"
      End
      Begin PaneConfiguration = 5
         NumPanes = 2
         Configuration = "(H (2 [66] 3))"
      End
      Begin PaneConfiguration = 6
         NumPanes = 2
         Configuration = "(H (4 [50] 3))"
      End
      Begin PaneConfiguration = 7
         NumPanes = 1
         Configuration = "(V (3))"
      End
      Begin PaneConfiguration = 8
         NumPanes = 3
         Configuration = "(H (1[56] 4[18] 2) )"
      End
      Begin PaneConfiguration = 9
         NumPanes = 2
         Configuration = "(H (1 [75] 4))"
      End
      Begin PaneConfiguration = 10
         NumPanes = 2
         Configuration = "(H (1[66] 2) )"
      End
      Begin PaneConfiguration = 11
         NumPanes = 2
         Configuration = "(H (4 [60] 2))"
      End
      Begin PaneConfiguration = 12
         NumPanes = 1
         Configuration = "(H (1) )"
      End
      Begin PaneConfiguration = 13
         NumPanes = 1
         Configuration = "(V (4))"
      End
      Begin PaneConfiguration = 14
         NumPanes = 1
         Configuration = "(V (2))"
      End
      ActivePaneConfig = 0
   End
   Begin DiagramPane = 
      Begin Origin = 
         Top = 0
         Left = 0
      End
      Begin Tables = 
         Begin Table = "replays"
            Begin Extent = 
               Top = 6
               Left = 38
               Bottom = 221
               Right = 208
            End
            DisplayFlags = 280
            TopColumn = 0
         End
         Begin Table = "p_replays"
            Begin Extent = 
               Top = 0
               Left = 519
               Bottom = 228
               Right = 689
            End
            DisplayFlags = 280
            TopColumn = 1
         End
      End
   End
   Begin SQLPane = 
   End
   Begin DataPane = 
      Begin ParameterDefaults = ""
      End
   End
   Begin CriteriaPane = 
      Begin ColumnWidths = 11
         Column = 1440
         Alias = 900
         Table = 1170
         Output = 720
         Append = 1400
         NewValue = 1170
         SortType = 1350
         SortOrder = 1410
         GroupBy = 1350
         Filter = 1350
         Or = 1350
         Or = 1350
         Or = 1350
      End
   End
End
' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'vwReplayAnalysis'
GO
EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPaneCount', @value=1 , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'vwReplayAnalysis'
GO
