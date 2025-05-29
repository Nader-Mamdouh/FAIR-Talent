using AutoMapper;
using FAIR.Application.DTOs.Identity;
using FAIR.Application.DTOs.Profile;
using FAIR.Application.DTOs.Report;
using FAIR.Domain.Entities;
using FAIR.Domain.Entities.Identity;
namespace FAIR.Application.Mapping
{
    public class MappingConfig : Profile
    {
        public MappingConfig()
        {
            CreateMap<CreateReport, Report>().ReverseMap();
            CreateMap<UpdateReport, Report>().ReverseMap();
            CreateMap<GetReport,Report>().ReverseMap();
            CreateMap<GetPlayer,Player>().ReverseMap();
            CreateMap <Player, Register>().ReverseMap();
            CreateMap<Coach, Register>().ReverseMap();
            CreateMap<Player, UpdatePlayerProfile>().ReverseMap();
            CreateMap<Player, PlayerProfile>().ReverseMap();
            CreateMap<Coach, UpdatePlayerProfile>().ReverseMap();
            CreateMap<Coach, CoachProfile>().ReverseMap();







        }
    }
}
