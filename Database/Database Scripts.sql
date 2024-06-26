USE [CUNY]
GO
/****** Object:  Table [dbo].[BCs]    Script Date: 5/15/2024 6:55:23 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[BCs](
	[index] [bigint] NULL,
	[t] [float] NULL,
	[BC] [int] NULL,
	[SelectedPC] [varchar](max) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[BCs_Archive]    Script Date: 5/15/2024 6:55:24 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[BCs_Archive](
	[index] [bigint] NULL,
	[t] [float] NULL,
	[BC] [int] NULL,
	[SelectedPC] [varchar](max) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Experiments_Archive]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  View [dbo].[vwSpike_PC_Times - All BCs]    Script Date: 5/15/2024 6:55:24 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO




CREATE VIEW [dbo].[vwSpike_PC_Times - All BCs]
AS
SELECT        [index], t, CAST(t AS int) AS time_ms, BC, expid
FROM            BCs
UNION ALL
SELECT        a.[index], a.t, CAST(a.t AS int) AS time_ms, a.BC, a.expid
FROM            BCs_Archive AS a INNER JOIN
                         Experiments_Archive AS b ON a.expid = b.ID
WHERE        (b.DeArchive = 1)
GO
/****** Object:  Table [dbo].[Experiments]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  View [dbo].[vwExperiments]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  View [dbo].[vwFailedExp]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  Table [dbo].[spike_times_Archive]    Script Date: 5/15/2024 6:55:24 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[spike_times_Archive](
	[index] [bigint] NULL,
	[0] [float] NULL,
	[SelectedPC] [varchar](max) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[spike_times]    Script Date: 5/15/2024 6:55:24 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[spike_times](
	[index] [bigint] NULL,
	[0] [float] NULL,
	[SelectedPC] [varchar](max) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  View [dbo].[vwSpikeTimes]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  Table [dbo].[spiking_neurons_Archive]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  Table [dbo].[spiking_neurons]    Script Date: 5/15/2024 6:55:24 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[spiking_neurons](
	[index] [bigint] NULL,
	[0] [int] NULL,
	[SelectedPC] [varchar](max) NULL,
	[expid] [bigint] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  View [dbo].[vwSpikingNeurons]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  Table [dbo].[SynWeightsStats_Archive]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  Table [dbo].[SynWeightsStats]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  View [dbo].[vwSynWeightsStats]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  View [dbo].[vwExperimentsAll]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  Table [dbo].[wexc_s_Archive]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  Table [dbo].[wexc_s]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  View [dbo].[vwSynapticChanges]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  View [dbo].[vwSpikingPCFilter]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  View [dbo].[vwSpike_PC_Times]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  View [dbo].[vwSpikesByExp]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  View [dbo].[vwSpike_PC_Times - All PCs]    Script Date: 5/15/2024 6:55:24 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO





CREATE VIEW [dbo].[vwSpike_PC_Times - All PCs]
AS
SELECT        a.[index], b.[0] AS [PC Num], a.[0] AS Spike_Time, CAST(a.[0] AS int) AS Spike_Time_int, a.expid, c.InputFromPC AS [PC Filter]
FROM            vwSpikeTimes AS a INNER JOIN
                         vwSpikingNeurons AS b ON a.[index] = b.[index] AND a.expid = b.expid LEFT OUTER JOIN
                         vwSpikingPCFilter AS c ON b.expid = c.expid AND b.[0] = c.InputFromPC					 
--where dbo.spike_times.expid in (select top (12) id from Experiments order by id desc)
						  
GO
/****** Object:  Table [dbo].[PSCs]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  Table [dbo].[PSCs_Archive]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  Table [dbo].[rate]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  Table [dbo].[rate_Archive]    Script Date: 5/15/2024 6:55:24 PM ******/
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
/****** Object:  StoredProcedure [dbo].[ArchiveExp]    Script Date: 5/15/2024 6:55:24 PM ******/
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
insert into [dbo].[PSCs_archive] select * from [dbo].[PSCs] where expid = (@ExpID)
insert into [dbo].[SynWeightsStats_archive] select * from [dbo].[SynWeightsStats] where expid = (@ExpID)
insert into [dbo].[BCs_archive] select * from [dbo].[BCs] where expid = (@ExpID)
insert into [dbo].[rate_archive] select * from [dbo].[rate] where expid = (@ExpID)
insert into [dbo].[spike_times_archive] select * from [dbo].[spike_times] where expid = (@ExpID)
insert into [dbo].[spiking_neurons_archive] select * from [dbo].[spiking_neurons] where expid = (@ExpID)
insert into [dbo].[wexc_s_archive] select * from [dbo].[wexc_s] where expid = (@ExpID)
insert into [dbo].[Experiments_archive] select *, null from [dbo].[Experiments] where id = (@ExpID)
exec [dbo].[CleanUpExp] @ExpID
Update [dbo].[Experiments_archive] set [DeArchive] = 0 where [ID] = @ExpID 
END
GO
/****** Object:  StoredProcedure [dbo].[CleanUp]    Script Date: 5/15/2024 6:55:24 PM ******/
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
delete from [dbo].[spike_times]
where expid in (select [id] from [dbo].[vwFailedExp])
delete from [dbo].[spiking_neurons]
where expid in (select [id] from [dbo].[vwFailedExp])
delete from [dbo].[wexc_s]
where expid in (select [id] from [dbo].[vwFailedExp])
delete from [dbo].[Experiments]
where id in (select [id] from [dbo].[vwFailedExp])
END
GO
/****** Object:  StoredProcedure [dbo].[CleanUpExp]    Script Date: 5/15/2024 6:55:24 PM ******/
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
delete from [dbo].[PSCs]
where expid = (@ExpID)
delete from [dbo].[SynWeightsStats]
where expid = (@ExpID)
delete from [dbo].[BCs]
where expid = (@ExpID)
delete from [dbo].[rate]
where expid = (@ExpID)
delete from [dbo].[spike_times]
where expid = (@ExpID)
delete from [dbo].[spiking_neurons]
where expid = (@ExpID)
delete from [dbo].[wexc_s]
where expid = (@ExpID)
delete from [dbo].[Experiments]
where id = (@ExpID)
END
GO
/****** Object:  StoredProcedure [dbo].[DeArchiveExp]    Script Date: 5/15/2024 6:55:24 PM ******/
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
