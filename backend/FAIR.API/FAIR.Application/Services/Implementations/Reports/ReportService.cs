using AutoMapper;
using FAIR.API.Interfaces.Report;
using FAIR.Application.DTOs;
using FAIR.Application.DTOs.Report;
using FAIR.Domain.Entities;
using FAIR.Domain.Interfaces;
namespace FAIR.Application.Services.Implementations.Reports
{
    public class ReportService
        ( IReportRepository _reportRepository , IMapper mapper)
        : IReportService
    {
        public async Task<ServiceResponse> AddAsync(CreateReport report)
        {
            var Mappingreport = mapper.Map<Report>(report);
            Mappingreport.CreatedDate = DateTime.Now;
            
            var result = await _reportRepository.AddAsync(Mappingreport);
            return result > 0 ? new ServiceResponse(true, "Report is Added!")
                : new ServiceResponse(true, "Fail to Add Report");
        }

        public async Task<ServiceResponse> DeleteAsync(string Id)
        {
            var result = await _reportRepository.DeleteAsync(Id);
            return result > 0 ? new ServiceResponse(true, $"Report with Id {Id} is Delete!")
                : new ServiceResponse(true, $"Fail to Delete This Report with Id {Id}");
        }

        public async Task<IEnumerable<GetReport>> GetAllAsync()
        {
            var reports = await _reportRepository.GetAllAsync();
            if (!reports.Any())
                return [];
            var result = mapper.Map<IEnumerable<GetReport>>(reports);
            return result;
        }

        public async Task<GetReport> GetByIdAsync(string id)
        {
            var report = await _reportRepository.GetByIdAsync(id);
            if (report == null)
                return new GetReport();
            var result = mapper.Map<GetReport>(report);
            return result;
        }
        public async Task<IEnumerable<GetReport>> GetReportsByPlayerIdAsync(string playerId)
        {
            var reports = await _reportRepository.GetReportsByPlayerIdAsync(playerId);
            if(!reports .Any())
                return [];
            var result = mapper.Map<IEnumerable<GetReport>>(reports);
            return result;
        }
        public async Task<IEnumerable<GetPlayer>> GetTopPlayersByScoreAsync(int count)
        {
            var topPlayers = await _reportRepository.GetTopPlayersByScoreAsync(count);
       
           if (!topPlayers .Any())
                return [];
            var result = mapper.Map<IEnumerable<GetPlayer>>(topPlayers);


            foreach(var res in result  )
            {
                res.AverageScore = await _reportRepository.AverageScorePercentage(res.Id!);
            }

            return result;
        }

        public async Task<ServiceResponse> UpdateAsync(UpdateReport report)
        {
            var mappingReport = mapper.Map<Report>(report);
            var resultUpdate = await _reportRepository.UpdateAsync(mappingReport);
            return resultUpdate > 0 ? new ServiceResponse(true, "Updated Report!")
                : new ServiceResponse(true, "Fail to Update Report");
        }
    }
}
